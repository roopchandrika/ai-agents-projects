"""
Tests for the LangGraph ticket router.

These build the REAL graph (StateGraph, conditional edges, compiled app)
and run it end-to-end, but with fake classify/draft functions substituted
for the Anthropic API calls -- so they test actual LangGraph routing and
state-accumulation behavior without needing an API key or network access.

Run with: python -m pytest test_graph.py -v
"""

import pytest

from graph import build_graph, run_ticket, route_by_category


def fake_classify(text: str) -> str:
    """Naive keyword classifier standing in for the LLM during tests."""
    lowered = text.lower()
    if "charge" in lowered or "refund" in lowered or "invoice" in lowered:
        return "billing"
    if "crash" in lowered or "bug" in lowered or "error" in lowered:
        return "technical"
    return "general"


def fake_draft(text: str, category: str) -> str:
    return f"[{category} draft] Thanks for reaching out about: {text[:30]}..."


@pytest.fixture
def app():
    return build_graph(fake_classify, fake_draft)


def test_billing_ticket_routes_to_billing_handler(app):
    result = run_ticket(app, "I was charged twice on my invoice.")
    assert result["category"] == "billing"
    assert "billing draft" in result["draft_response"]


def test_technical_ticket_routes_to_technical_handler(app):
    result = run_ticket(app, "The app keeps crashing on export.")
    assert result["category"] == "technical"
    assert "technical draft" in result["draft_response"]


def test_general_ticket_routes_to_general_handler(app):
    result = run_ticket(app, "What are your support hours?")
    assert result["category"] == "general"
    assert "general draft" in result["draft_response"]


def test_trace_records_full_path_through_graph(app):
    result = run_ticket(app, "My card was double charged.")
    assert len(result["trace"]) == 2
    assert "classified as 'billing'" in result["trace"][0]
    assert "drafted response via 'billing' handler" in result["trace"][1]


def test_route_by_category_falls_back_to_general_for_unknown_category():
    # Defends against a classifier ever returning something outside the
    # three known categories -- routing must not raise, it must degrade
    # to "general" instead.
    fake_state = {"ticket_text": "x", "category": "not_a_real_category", "draft_response": "", "trace": []}
    assert route_by_category(fake_state) == "general"


def test_full_graph_never_raises_on_odd_classifier_output():
    def weird_classify(text: str) -> str:
        return "SOMETHING_UNEXPECTED"

    app = build_graph(weird_classify, fake_draft)
    result = run_ticket(app, "anything")
    assert result["category"] == "general"
