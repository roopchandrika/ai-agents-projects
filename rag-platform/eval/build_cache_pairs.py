"""Generate threshold-tuning pairs for the semantic cache from the eval questions.

For each question: a paraphrase that must hit the cache (same answer) and a near-miss that must NOT (reads
almost the same but needs a different answer). Near-misses are the hard part: they're what makes a threshold
too low return confidently wrong cached answers.

Usage: python -m eval.build_cache_pairs     -> eval/cache_pairs.jsonl (review before tuning)
"""
import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from rag import config
from rag.pipeline import claude

PAIRS_PATH = config.EVAL_DIR / "cache_pairs.jsonl"
STRATEGIES = ["swap the main entity for a similar one (another mission, person, spacecraft or agency)",
              "keep the entity but ask about a different attribute (e.g. mass instead of power, cost instead of date)",
              "change a qualifier (first vs last, launch vs landing, a different year or number)"]

PROMPT = """Here is a question a user asked a space-exploration Q&A system:

<question>{question}</question>

Write two new questions.

1. paraphrase: the same question worded differently, the way another user might type it. It must have exactly the \
same answer. Vary the wording and structure; don't just reorder words.
2. near_miss: a question that reads almost the same as the original (reuse most of its words) but needs a \
different answer. Do this: {strategy}. It must be a real, sensible question, not a trick.

Also say in a few words why the near-miss has a different answer."""


class Pair(BaseModel):
    paraphrase: str
    near_miss: str
    why_different: str


def make_pair(args) -> Pair:
    question, strategy = args
    return claude().messages.parse(
        model=config.EVAL_GEN_MODEL, max_tokens=16000, output_format=Pair,
        messages=[{"role": "user", "content": PROMPT.format(question=question, strategy=strategy)}],
    ).parsed_output


def main() -> None:
    items = [json.loads(line) for line in (config.EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    jobs = [(it["question"], STRATEGIES[i % len(STRATEGIES)]) for i, it in enumerate(items)]
    print(f"Generating {len(jobs)} pairs with {config.EVAL_GEN_MODEL}...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(make_pair, jobs))
    with PAIRS_PATH.open("w", encoding="utf-8") as f:
        for it, (_, strategy), p in zip(items, jobs, pairs):
            f.write(json.dumps({"id": it["id"], "question": it["question"], "paraphrase": p.paraphrase,
                                "near_miss": p.near_miss, "why_different": p.why_different,
                                "strategy": strategy.split(" (")[0]}, ensure_ascii=False) + "\n")
    print(f"Wrote {PAIRS_PATH}")


if __name__ == "__main__":
    main()
