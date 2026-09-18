# function-calling-agent

A minimal agent built directly on **Anthropic's native tool-use (function-calling) API** —
no agent framework (no LangChain, no CrewAI). It demonstrates the core
tool-calling loop: the model decides to call a tool, the tool executes locally,
the result is fed back to the model, and this repeats until the model is ready
to answer in plain text.

## What it does

The agent has two tools:

| Tool | Purpose |
|---|---|
| `calculator` | Evaluates a basic arithmetic expression (`+ - * / // % **`, parentheses) |
| `unit_converter` | Converts a value between miles/km, lb/kg, °F/°C, and feet/meters |

Example:

```
$ python agent.py "What is 42 * 17, and how many kilometers is 26.2 miles?"
  -> tool call: calculator({'expression': '42 * 17'})
     [ok] {"result": 714}
  -> tool call: unit_converter({'value': 26.2, 'from_unit': 'miles', 'to_unit': 'kilometers'})
     [ok] {"result": 42.164708}
42 * 17 = 714. And 26.2 miles is approximately 42.16 kilometers.
```

The agent supports **multi-turn tool use**: a single user query can trigger
several tool calls in sequence (as above) before the model produces its final
answer, capped at `MAX_TURNS` to avoid runaway loops.

## Why native tool-calling (not a framework)

This project intentionally uses the Anthropic SDK's tool-use API directly
rather than a framework abstraction. The goal was to understand exactly how
the reason -> act -> observe loop works underneath frameworks like LangChain:
how tool schemas are defined, how `stop_reason == "tool_use"` signals that the
model wants to call a tool, and how `tool_result` blocks (including the
`is_error` flag) get fed back into the conversation.

## Error handling

Malformed or invalid tool calls do **not** crash the loop. `execute_tool()`
catches:
- Unknown tool names
- Missing/invalid arguments (e.g. a non-numeric `value`, an unsupported unit pair)
- Runtime errors inside a tool (e.g. division by zero)

...and converts them into a `tool_result` with `is_error: true`. Claude sees
the error message and can retry with corrected arguments or explain the
failure to the user, rather than the whole program throwing an exception.

The calculator also **whitelists characters** before evaluating an expression,
rather than calling `eval()` on arbitrary model output, to avoid arbitrary
code execution if the model (or a prompt injected into its context) ever
produced a malicious expression.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your key
export ANTHROPIC_API_KEY=sk-ant-...   # or use python-dotenv / your shell profile
```

## Usage

```bash
# One-shot query
python agent.py "Convert 100 fahrenheit to celsius"

# Interactive mode
python agent.py
```

## Tests

Unit tests cover the tool implementations and the error-handling contract
(unknown tool / malformed input -> `is_error` result, not an exception).
They run without an API key:

```bash
pip install pytest
python -m pytest test_agent.py -v
```

## What I'd add next

- A `pytest`-based integration test that hits the real API with a cassette
  (e.g. `vcrpy`) to test the full loop against real model behavior, not just
  the mocked version used in development.
- A third tool (e.g. a date/time calculator) to exercise tool *selection*
  (does the model pick the right tool when more than two are available?),
  not just multi-turn chaining between two.
- Streaming support (`client.messages.stream`) for lower perceived latency
  in interactive mode.

## Project structure

```
function-calling-agent/
├── agent.py          # tool schemas, tool implementations, agent loop, CLI
├── test_agent.py      # unit tests for tools + error handling (no API key needed)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
