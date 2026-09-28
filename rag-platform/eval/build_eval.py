"""Draft eval questions with Claude: one question per sampled chunk, with a reference answer.

Usage: python -m eval.build_eval [--n 80] [--seed 42]
Writes eval/questions.draft.jsonl. Review it with `python -m eval.review_eval` before using it:
generated questions are sometimes ambiguous, trivial or answerable from several documents.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from rag import config
from rag.corpus import chunk_all, load_documents
from rag.pipeline import claude
from rag.store import embedder

DRAFT_PATH = config.EVAL_DIR / "questions.draft.jsonl"

PROMPT = """Here is a passage from the Wikipedia article "{title}":

<passage>
{text}
</passage>

Write one question that a curious person might ask a space-exploration Q&A assistant, which this passage answers.

- The question must make sense on its own, without seeing the passage: name the mission, person or thing it is \
about, and never say "the passage" or "the article".
- It should have one clear, checkable answer (a fact, number, date, name or short explanation), not an opinion.
- Prefer specific details over the article's opening summary.

Then give the reference answer in one or two sentences, using only the passage."""


class QAPair(BaseModel):
    question: str
    reference_answer: str


def draft(chunk) -> QAPair:
    response = claude().messages.parse(
        model=config.EVAL_GEN_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": PROMPT.format(title=chunk.title, text=chunk.text)}],
        output_format=QAPair,
    )
    return response.parsed_output


def sample_chunks(n: int, seed: int) -> list:
    """Spread questions across documents: cycle through documents in random order,
    picking one substantial chunk from each, so no single article dominates."""
    rng = random.Random(seed)
    docs = load_documents()
    by_doc: dict[str, list] = {}
    for c in chunk_all(docs, embedder().tokenizer):
        if len(c.text) >= 800:
            by_doc.setdefault(c.doc_id, []).append(c)
    order = list(by_doc)
    rng.shuffle(order)
    picked, used = [], set()
    while len(picked) < n:
        for doc_id in order:
            options = [c for c in by_doc[doc_id] if c.chunk_id not in used]
            if options and len(picked) < n:
                c = rng.choice(options)
                used.add(c.chunk_id)
                picked.append(c)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80, help="questions to draft (review will drop some)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = sample_chunks(args.n, args.seed)
    print(f"Drafting {len(chunks)} questions with {config.EVAL_GEN_MODEL}...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(draft, chunks))

    with DRAFT_PATH.open("w", encoding="utf-8") as f:
        for i, (c, qa) in enumerate(zip(chunks, pairs), start=1):
            f.write(json.dumps({
                "id": f"q{i:03d}", "question": qa.question, "reference_answer": qa.reference_answer,
                "doc_id": c.doc_id, "chunk_id": c.chunk_id, "title": c.title, "source_text": c.text,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {DRAFT_PATH}. Next: python -m eval.review_eval")


if __name__ == "__main__":
    main()
