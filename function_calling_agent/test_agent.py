"""
Unit tests for the tool implementations in agent.py.

These test the tools directly and do not call the Anthropic API, so they
run without an API key -- e.g. in CI.

Run with:  python -m pytest test_agent.py -v
"""

import json

import pytest

from agent import execute_tool, run_calculator, run_unit_converter, ToolError


# --- calculator -------------------------------------------------------------

def test_calculator_basic_arithmetic():
    assert run_calculator("2 + 2") == 4
    assert run_calculator("42 * 17") == 714
    assert run_calculator("(3 + 4) / 2") == 3.5


def test_calculator_rejects_disallowed_characters():
    with pytest.raises(ToolError):
        run_calculator("__import__('os').system('echo hi')")


def test_calculator_rejects_empty_expression():
    with pytest.raises(ToolError):
        run_calculator("")


def test_calculator_division_by_zero():
    with pytest.raises(ToolError):
        run_calculator("1 / 0")


# --- unit_converter -----------------------------------------------------------

def test_unit_converter_miles_to_km():
    assert run_unit_converter(26.2, "miles", "kilometers") == pytest.approx(42.164, rel=1e-3)


def test_unit_converter_fahrenheit_to_celsius():
    assert run_unit_converter(32, "fahrenheit", "celsius") == pytest.approx(0.0)


def test_unit_converter_same_unit_passthrough():
    assert run_unit_converter(10, "meters", "meters") == 10


def test_unit_converter_unsupported_pair_raises():
    with pytest.raises(ToolError):
        run_unit_converter(10, "miles", "kilograms")


# --- execute_tool (the dispatcher the agent loop calls) ----------------------

def test_execute_tool_success_shape():
    outcome = execute_tool("calculator", {"expression": "5 * 5"})
    assert outcome["is_error"] is False
    assert json.loads(outcome["content"])["result"] == 25


def test_execute_tool_unknown_tool_returns_error_not_exception():
    outcome = execute_tool("not_a_real_tool", {})
    assert outcome["is_error"] is True
    assert "Unknown tool" in outcome["content"]


def test_execute_tool_malformed_input_returns_error_not_exception():
    # Missing required "expression" key -- should surface as a tool error,
    # not crash the agent loop.
    outcome = execute_tool("calculator", {})
    assert outcome["is_error"] is True
