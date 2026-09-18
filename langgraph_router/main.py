"""
main.py
-------
Runs one or more support tickets through the live LangGraph workflow
(real Anthropic API calls for classification + drafting).

Usage:
    python main.py "My card was charged twice this month, please help."
    python main.py                      # runs 3 built-in example tickets
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from graph import build_live_graph, run_ticket

EXAMPLE_TICKETS = [
    "I was charged $49.99 twice on my last invoice, can you refund the duplicate?",
    "The app crashes every time I try to export a PDF on version 4.2.1.",
    "What are your support hours on weekends?",
]


def print_result(ticket_text: str, result: dict) -> None:
    print("=" * 70)
    print(f"TICKET:   {ticket_text}")
    print(f"CATEGORY: {result['category']}")
    print(f"TRACE:    {' -> '.join(result['trace'])}")
    print(f"RESPONSE:\n{result['draft_response']}")
    print()


def main() -> None:
    app = build_live_graph()

    if len(sys.argv) > 1:
        ticket_text = " ".join(sys.argv[1:])
        print_result(ticket_text, run_ticket(app, ticket_text))
        return

    for ticket_text in EXAMPLE_TICKETS:
        print_result(ticket_text, run_ticket(app, ticket_text))


if __name__ == "__main__":
    main()
