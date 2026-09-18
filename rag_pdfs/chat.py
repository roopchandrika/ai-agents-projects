"""
chat.py
-------
The RAG query loop: embed the user's question, retrieve the most relevant
chunks from the PDF corpus, and ask Claude to answer grounded in that
context, citing which document/page each part of the answer came from.

Usage:
    python ingest.py sample_docs/*.pdf     # run once to build the index
    python chat.py "What does the document say about X?"
    python chat.py                         # interactive mode
"""

import os
import sys

from anthropic import Anthropic

from retrieve import build_context, load_collection, retrieve

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context, which is drawn from a set of PDF documents. Each context
block is tagged with its source, like [filename.pdf, p.3].

Rules:
- Cite the source and page for every specific claim, using the same
  [filename.pdf, p.N] format shown in the context.
- If the context does not contain the answer, say so explicitly instead of
  guessing or using outside knowledge.
- Keep the answer concise."""


def answer_question(client: Anthropic, question: str, collection, embedder, top_k: int = 3) -> str:
    chunks = retrieve(question, collection, embedder, top_k=top_k)
    if not chunks:
        return "No indexed documents to search. Run ingest.py first."

    context = build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in your environment (see .env.example).", file=sys.stderr)
        sys.exit(1)

    try:
        collection, embedder = load_collection()
    except Exception as exc:
        print(f"Could not load the index ({exc}). Run ingest.py first, e.g.:\n"
              f"  python ingest.py sample_docs/*.pdf", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(answer_question(client, question, collection, embedder))
        return

    print("RAG PDF chatbot. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(f"bot> {answer_question(client, question, collection, embedder)}")


if __name__ == "__main__":
    main()
