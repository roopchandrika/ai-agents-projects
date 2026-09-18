"""
graph.py
--------
A stateful, graph-based support-ticket router built with LangGraph.

Flow:

    START -> classify -> (conditional branch on category) -> draft -> END
                              |-> billing
                              |-> technical
                              |-> general

The "classify" node predicts a category and writes it into shared state.
A conditional edge function reads that category and routes to one of three
handler nodes, each of which drafts a resolution using a category-specific
system prompt. Every node appends to a `trace` list in state, so the full
path a ticket took through the graph is inspectable afterwards -- this is
the "stateful" part LangGraph adds over a plain function-calling loop: state
persists and accumulates across nodes, and branching is explicit in the
graph structure rather than buried in if/else code.
"""

from typing import Callable, TypedDict

from langgraph.graph import StateGraph, START, END

from llm import VALID_CATEGORIES, classify_ticket_llm, draft_response_llm


class TicketState(TypedDict):
    ticket_text: str
    category: str
    draft_response: str
    trace: list[str]


# ---------------------------------------------------------------------------
# Node functions
#
# Each node takes the injected `classify_fn` / `draft_fn` rather than calling
# the Anthropic API directly, so tests can substitute fake functions and
# exercise the graph's routing/state logic with zero network calls.
# ---------------------------------------------------------------------------

def make_classify_node(classify_fn: Callable[[str], str]):
    def classify(state: TicketState) -> dict:
        category = classify_fn(state["ticket_text"])
        if category not in VALID_CATEGORIES:
            category = "general"
        return {
            "category": category,
            "trace": state["trace"] + [f"classified as '{category}'"],
        }

    return classify


def make_handler_node(category: str, draft_fn: Callable[[str, str], str]):
    def handle(state: TicketState) -> dict:
        response = draft_fn(state["ticket_text"], category)
        return {
            "draft_response": response,
            "trace": state["trace"] + [f"drafted response via '{category}' handler"],
        }

    return handle


def route_by_category(state: TicketState) -> str:
    """Conditional-edge function: reads state, returns the next node's name."""
    return state["category"] if state["category"] in VALID_CATEGORIES else "general"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(classify_fn: Callable[[str], str], draft_fn: Callable[[str, str], str]):
    graph = StateGraph(TicketState)

    graph.add_node("classify", make_classify_node(classify_fn))
    for category in VALID_CATEGORIES:
        graph.add_node(category, make_handler_node(category, draft_fn))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {category: category for category in VALID_CATEGORIES},
    )
    for category in VALID_CATEGORIES:
        graph.add_edge(category, END)

    return graph.compile()


def build_live_graph():
    """Build the graph wired to the real Anthropic API (used by main.py)."""
    from llm import get_client

    client = get_client()
    classify_fn = lambda text: classify_ticket_llm(client, text)
    draft_fn = lambda text, category: draft_response_llm(client, text, category)
    return build_graph(classify_fn, draft_fn)


def run_ticket(app, ticket_text: str) -> TicketState:
    initial_state: TicketState = {
        "ticket_text": ticket_text,
        "category": "",
        "draft_response": "",
        "trace": [],
    }
    return app.invoke(initial_state)
