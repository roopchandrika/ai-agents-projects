"""Run every eval question through the pipeline and score it.

Usage: python -m eval.run_eval --label baseline [--limit 10] [--workers 4]

Per question it records retrieval hits (was the source document / exact source chunk in the top k?),
an LLM-judge correctness verdict, citations, tokens, cost and latency. Writes
eval/results/<run_id>.csv and appends one row to eval/results/summary.csv.
"""
import argparse
import csv
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pydantic import BaseModel

from rag import config, pipeline
from rag.store import qdrant

QUESTIONS_PATH = config.EVAL_DIR / "questions.jsonl"
SUMMARY_PATH = config.RESULTS_DIR / "summary.csv"

JUDGE_PROMPT = """You are grading an answer from a question-answering system against a reference answer.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<candidate_answer>{candidate}</candidate_answer>

The candidate is correct if it states the key facts of the reference answer and does not contradict it. \
Extra detail is fine, and so are wording differences and citation markers like [1]. It is incorrect if it misses \
or gets wrong the key fact, or says it doesn't know. Explain your reasoning briefly, then give the verdict."""


class Verdict(BaseModel):
    reasoning: str
    correct: bool


def judge(question: str, reference: str, candidate: str) -> tuple[Verdict, float]:
    response = pipeline.claude().messages.parse(
        model=config.JUDGE_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, candidate=candidate)}],
        output_format=Verdict,
    )
    cost = config.cost_usd(config.JUDGE_MODEL, response.usage.input_tokens, response.usage.output_tokens)
    return response.parsed_output, cost


def evaluate(item: dict) -> dict:
    result = pipeline.answer(item["question"])
    retrieved_docs = [h.doc_id for h in result.sources]
    verdict, judge_cost = judge(item["question"], item["reference_answer"], result.answer)
    return {
        "id": item["id"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "answer": result.answer,
        "correct": int(verdict.correct),
        "judge_reasoning": verdict.reasoning,
        "doc_hit": int(item["doc_id"] in retrieved_docs),
        "chunk_hit": int(item["chunk_id"] in [h.chunk_id for h in result.sources]),
        "cited": int(bool(result.cited_doc_ids)),
        "cited_gold_doc": int(item["doc_id"] in result.cited_doc_ids),
        "gold_doc": item["doc_id"],
        "retrieved_docs": json.dumps(retrieved_docs),
        "model": result.model,
        "stop_reason": result.stop_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "judge_cost_usd": round(judge_cost, 6),
        "retrieval_ms": round(result.retrieval_ms, 1),
        "generation_ms": round(result.generation_ms, 1),
        "total_ms": round(result.total_ms, 1),
    }


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1] if len(values) > 1 else values[0]


def summarize(run_id: str, label: str, rows: list[dict]) -> dict:
    n = len(rows)
    latencies = [r["total_ms"] for r in rows]
    mean = lambda key: round(sum(r[key] for r in rows) / n, 4)
    return {
        "run_id": run_id, "label": label, "n": n,
        "answer_accuracy": mean("correct"),
        "doc_hit_rate": mean("doc_hit"),
        "chunk_hit_rate": mean("chunk_hit"),
        "citation_rate": mean("cited"),
        "p50_ms": round(percentile(latencies, 50)),
        "p95_ms": round(percentile(latencies, 95)),
        "cost_per_request_usd": round(mean("cost_usd"), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in rows), 4),
        "answer_model": config.ANSWER_MODEL, "judge_model": config.JUDGE_MODEL,
        "embed_model": config.EMBED_MODEL, "top_k": config.TOP_K,
        "chunk_tokens": config.CHUNK_TOKENS, "chunk_overlap": config.CHUNK_OVERLAP,
    }


def write_csv(path, rows: list[dict], append: bool = False) -> None:
    new_file = not path.exists() or not append
    with path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="name for this run, e.g. baseline or cache-v1")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--limit", type=int, help="only run the first N questions (quick checks)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    items = items[:args.limit] if args.limit else items
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}"
    print(f"Running {len(items)} questions ({run_id})...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(evaluate, items), start=1):
            rows.append(row)
            print(f"  {i:>3}/{len(items)} {'OK ' if row['correct'] else 'BAD'} "
                  f"doc_hit={row['doc_hit']} {row['total_ms']:>6.0f}ms  {row['question'][:70]}")
    qdrant().close()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(config.RESULTS_DIR / f"{run_id}.csv", rows)
    summary = summarize(run_id, args.label, rows)
    write_csv(SUMMARY_PATH, [summary], append=True)

    print("\n" + "\n".join(f"{k:>22}: {v}" for k, v in summary.items()))
    print(f"\nPer-question results: eval/results/{run_id}.csv")


if __name__ == "__main__":
    main()
