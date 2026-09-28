"""Download the HotpotQA dev set (distractor setting) and sample questions for the GraphRAG project.

Usage: python -m scripts.fetch_hotpotqa [--n 150] [--seed 13]

The full dev set comes from the hotpotqa organisation's copy on Hugging Face (27.5 MB Parquet; the CMU link on
hotpotqa.github.io was unreachable) and is saved to data/hotpotqa/raw/, which is not committed.
The sample, which is committed, keeps each question's answer, type, supporting facts and all 10 context
paragraphs (2 gold + 8 distractors). The corpus is the union of those paragraphs.
"""
import argparse
import json
import random

import httpx
import pandas as pd

from rag import config

URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/"
       "validation-00000-of-00001.parquet")
DIR = config.ROOT / "data" / "hotpotqa"
RAW = DIR / "raw" / "hotpot_dev_distractor.parquet"
SAMPLE = DIR / "sample.json"


def download() -> None:
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with httpx.stream("GET", URL, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with RAW.open("wb") as f:
            for block in r.iter_bytes():
                f.write(block)
    print(f"Saved {RAW.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    download()
    data = pd.read_parquet(RAW).to_dict("records")
    sample = random.Random(args.seed).sample(data, args.n)
    # Convert Hugging Face's column layout back to the original format: [[title, sent_id], ...] and
    # [[title, [sentences]], ...].
    out = [{"id": q["id"], "question": q["question"], "answer": q["answer"], "type": q["type"],
            "level": q["level"],
            "supporting_facts": [[t, int(i)] for t, i in zip(q["supporting_facts"]["title"],
                                                              q["supporting_facts"]["sent_id"])],
            "context": [[t, [str(x) for x in sents]] for t, sents in zip(q["context"]["title"],
                                                                        q["context"]["sentences"])]}
           for q in sample]
    SAMPLE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    titles = {title for q in out for title, _ in q["context"]}
    types = {t: sum(q["type"] == t for q in out) for t in ("bridge", "comparison")}
    print(f"Sampled {len(out)} questions {types}; corpus of {len(titles)} unique paragraphs -> {SAMPLE}")


if __name__ == "__main__":
    main()
