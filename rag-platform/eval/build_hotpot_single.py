"""Draft single-hop questions over the HotpotQA corpus, to check that GraphRAG doesn't hurt easy questions.

Every HotpotQA question needs two paragraphs, so this writes questions answerable from one gold paragraph, with a
short-span answer so exact match / F1 apply. Output uses the sample's format, so hotpot_eval can run on it.

Usage: python -m eval.build_hotpot_single [--n 40]   -> data/hotpotqa/single_hop.json (review before use)
Then:  python -m eval.hotpot_eval --mode vector --set single
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from graphrag.corpus import gold_titles, paragraphs, questions
from rag import config
from rag.pipeline import claude

OUT = config.HOTPOT_DIR / "single_hop.json"

PROMPT = """Here is a Wikipedia paragraph.

<paragraph title="{title}">
{text}
</paragraph>

Write one factual question answerable from this paragraph alone. Name the subject explicitly; never say "the
paragraph". The answer must be a short span copied from the paragraph: a name, date, number or place (not a
sentence). Return the question and that answer span."""


class Single(BaseModel):
    question: str
    answer: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    titles = sorted({t for q in questions() for t in gold_titles(q)})
    picked = random.Random(args.seed).sample(titles, args.n)
    paras = paragraphs()

    def draft(title: str) -> dict:
        p = paras[title]
        s = claude().messages.parse(model=config.EVAL_GEN_MODEL, max_tokens=4096, output_format=Single,
                                    messages=[{"role": "user", "content": PROMPT.format(title=title, text=p.text)}]
                                    ).parsed_output
        return {"id": f"single-{titles.index(title)}", "question": s.question, "answer": s.answer, "type": "single",
                "level": "easy", "supporting_facts": [[title, 0]], "context": [[title, list(p.sentences)]]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(draft, picked))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(items)} single-hop questions to {OUT}. Review them before running evals.")


if __name__ == "__main__":
    main()
