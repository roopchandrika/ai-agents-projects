"""
llm.py
------
Thin wrapper around the Anthropic API for the two model calls this workflow
needs: classifying a ticket, and drafting a resolution for it.

Kept separate from graph.py so the graph's routing and state-transition
logic can be unit-tested with a fake/mocked version of these two functions,
without needing a live API key.
"""

import os

from anthropic import Anthropic

MODEL = "claude-sonnet-4-5"

VALID_CATEGORIES = ("billing", "technical", "general")

_CLASSIFY_SYSTEM_PROMPT = f"""You are a support-ticket triage classifier.
Read the ticket text and respond with EXACTLY ONE WORD: one of
{", ".join(VALID_CATEGORIES)}.
No punctuation, no explanation, no extra words -- just the single category word."""

_DRAFT_SYSTEM_PROMPTS = {
    "billing": (
        "You are a billing support agent. Draft a short, polite resolution "
        "for the customer's billing issue. If you need account-specific data "
        "you don't have (e.g. their invoice number), say what you'd need to "
        "look up rather than inventing numbers."
    ),
    "technical": (
        "You are a technical support agent. Draft a short, clear resolution "
        "or troubleshooting step for the customer's technical issue. Ask a "
        "clarifying question if the ticket is too vague to act on."
    ),
    "general": (
        "You are a general support agent. Draft a short, friendly reply "
        "addressing the customer's question or routing them appropriately."
    ),
}


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=api_key)


def classify_ticket_llm(client: Anthropic, ticket_text: str) -> str:
    """Classify a ticket into one of VALID_CATEGORIES using Claude.

    Falls back to "general" if the model responds with something outside
    the expected set, so a routing decision is always well-formed even if
    the model output is unexpected.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        system=_CLASSIFY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket_text}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip().lower()
    category = raw.strip(".,! ")
    return category if category in VALID_CATEGORIES else "general"


def draft_response_llm(client: Anthropic, ticket_text: str, category: str) -> str:
    system_prompt = _DRAFT_SYSTEM_PROMPTS.get(category, _DRAFT_SYSTEM_PROMPTS["general"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": ticket_text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
