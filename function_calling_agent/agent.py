"""
function-calling-agent
-----------------------
A minimal agent built on Anthropic's native tool-use (function-calling) API.

It exposes two tools:
  - calculator: evaluates a basic arithmetic expression
  - unit_converter: converts a numeric value between a fixed set of units

The agent handles multi-turn tool invocation: Claude can call a tool, receive
the result, and decide whether to call another tool or respond to the user,
looping until it produces a final text answer.

Usage:
    python agent.py "What is 42 * 17, and how many kilometers is 26.2 miles?"
    python agent.py            # interactive mode
"""

import json
import os
import sys
from typing import Any

from anthropic import Anthropic, APIError

MODEL = "claude-sonnet-4-5"
MAX_TURNS = 6  # safety cap on the reason-act loop


# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema, per Anthropic's tool-use spec)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression using +, -, *, /, //, %, ** "
            "and parentheses. Use this any time the user asks for a numeric "
            "calculation instead of computing it yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '42 * 17' or '(3 + 4) / 2'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "unit_converter",
        "description": (
            "Convert a numeric value from one unit to another. Supported units: "
            "miles<->kilometers, pounds<->kilograms, fahrenheit<->celsius, "
            "feet<->meters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value to convert."},
                "from_unit": {
                    "type": "string",
                    "enum": ["miles", "kilometers", "pounds", "kilograms", "fahrenheit", "celsius", "feet", "meters"],
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["miles", "kilometers", "pounds", "kilograms", "fahrenheit", "celsius", "feet", "meters"],
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_ALLOWED_CHARS = set("0123456789+-*/.() %")


class ToolError(Exception):
    """Raised when a tool receives bad input or fails to execute."""


def run_calculator(expression: str) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("`expression` must be a non-empty string.")
    # Whitelist characters to avoid arbitrary code execution via eval().
    stripped = expression.replace(" ", "")
    if not set(stripped) <= (_ALLOWED_CHARS | {"*"}):
        raise ToolError(f"Expression contains disallowed characters: {expression!r}")
    try:
        # eval is safe here only because we've whitelisted characters above
        # and pass empty builtins.
        result = eval(stripped, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    except Exception as exc:
        raise ToolError(f"Could not evaluate expression {expression!r}: {exc}")
    if not isinstance(result, (int, float)):
        raise ToolError("Expression did not evaluate to a number.")
    return result


_CONVERSIONS = {
    ("miles", "kilometers"): lambda v: v * 1.60934,
    ("kilometers", "miles"): lambda v: v / 1.60934,
    ("pounds", "kilograms"): lambda v: v * 0.453592,
    ("kilograms", "pounds"): lambda v: v / 0.453592,
    ("feet", "meters"): lambda v: v * 0.3048,
    ("meters", "feet"): lambda v: v / 0.3048,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
}


def run_unit_converter(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    key = (from_unit, to_unit)
    if key not in _CONVERSIONS:
        raise ToolError(f"Unsupported conversion: {from_unit} -> {to_unit}")
    return _CONVERSIONS[key](value)


TOOL_IMPLEMENTATIONS = {
    "calculator": lambda inp: run_calculator(inp.get("expression")),
    "unit_converter": lambda inp: run_unit_converter(
        inp.get("value"), inp.get("from_unit"), inp.get("to_unit")
    ),
}


def execute_tool(name: str, tool_input: dict) -> dict:
    """Execute a tool call and return a well-formed tool_result payload.

    Malformed or unknown calls are converted into an `is_error` tool_result
    instead of raising, so a single bad tool call doesn't crash the agent
    loop -- Claude sees the error and can retry or explain it to the user.
    """
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"is_error": True, "content": f"Unknown tool: {name!r}"}
    try:
        result = impl(tool_input)
        return {"is_error": False, "content": json.dumps({"result": result})}
    except ToolError as exc:
        return {"is_error": True, "content": str(exc)}
    except Exception as exc:  # last-resort guard against unexpected failures
        return {"is_error": True, "content": f"Unexpected error in {name}: {exc}"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(client: Anthropic, user_message: str, verbose: bool = True) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TURNS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )
        except APIError as exc:
            return f"[agent error] Anthropic API call failed: {exc}"

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Claude produced a final answer -- collect and return the text.
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  -> tool call: {block.name}({block.input})")
            outcome = execute_tool(block.name, block.input)
            if verbose:
                tag = "ERROR" if outcome["is_error"] else "ok"
                print(f"     [{tag}] {outcome['content']}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": outcome["content"],
                    "is_error": outcome["is_error"],
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "[agent error] Reached max turns without a final answer."


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in your environment (see .env.example).", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_agent(client, query))
        return

    print("Function-calling agent (Anthropic). Type 'exit' to quit.")
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        print(f"agent> {run_agent(client, query)}")


if __name__ == "__main__":
    main()
