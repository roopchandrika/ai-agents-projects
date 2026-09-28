"""Evaluate retrieval modes on the HotpotQA sample.

Usage: python -m eval.hotpot_eval --mode vector [--retrieval-only] [--limit N]
       modes: vector, vector_rerank, graph, graph_decompose

Metrics (official HotpotQA answer normalization):
  EM / F1            short answer vs gold answer
  sp_recall          share of the gold supporting paragraphs present in the context given to the model
  all_gold           both supporting paragraphs present (the question is answerable from the context)
--retrieval-only skips the model and reports sp_recall / all_gold only, at no API cost.
Results are split by question type (bridge vs comparison). Appends to eval/results/hotpot_summary.csv.
"""
import argparse
import collections
import csv
import json
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from graphrag import answer as ans
from graphrag.corpus import gold_titles, questions
from graphrag import retrieve as retrieve_mod
from graphrag.retrieve import retrieve
from rag import config
from rag.store import qdrant

SUMMARY_PATH = config.RESULTS_DIR / "hotpot_summary.csv"
SUMMARY_FIELDS = ["run_id", "mode", "n", "retrieval_only"] + [
    f"{subset}_{metric}" for subset in ("all", "bridge", "comparison")
    for metric in ("sp_recall", "all_gold", "em", "f1")] + ["cost_per_question_usd", "p50_ms", "answer_model",
                                                             "rerank_model", "graph_strategy"]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = sum((collections.Counter(pt) & collections.Counter(gt)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall)


def evaluate(q: dict, mode: str, retrieval_only: bool, graph, linker) -> dict:
    t0 = time.perf_counter()
    cost, prediction = 0.0, ""
    if mode == "graph_decompose" and not retrieval_only:
        prediction, cost, r = ans.decomposed_answer(q["question"], graph, linker)
    else:
        r = retrieve(q["question"], "graph" if mode == "graph_decompose" else mode, graph, linker)
        if not retrieval_only:
            prediction, cost = ans.answer(q["question"], r)
    gold = gold_titles(q)
    found = gold & set(r.titles)
    return {
        "id": q["id"], "type": q["type"], "question": q["question"], "gold_answer": q["answer"],
        "prediction": prediction,
        "em": "" if retrieval_only else int(exact_match(prediction, q["answer"])),
        "f1": "" if retrieval_only else round(f1(prediction, q["answer"]), 4),
        "sp_recall": round(len(found) / len(gold), 4), "all_gold": int(found == gold),
        "context": json.dumps(r.titles), "linked_entities": len(r.linked_entities),
        "graph_candidates": r.graph_candidates, "triples": len(r.triples),
        "cost_usd": round(cost, 6), "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def summarize(run_id: str, mode: str, rows: list[dict], retrieval_only: bool) -> dict:
    out = {"run_id": run_id, "mode": mode, "n": len(rows), "retrieval_only": int(retrieval_only)}
    for name, subset in (("all", rows), ("bridge", [r for r in rows if r["type"] == "bridge"]),
                         ("comparison", [r for r in rows if r["type"] == "comparison"])):
        if not subset:
            continue
        mean = lambda k: round(sum(float(r[k]) for r in subset) / len(subset), 4)
        out[f"{name}_sp_recall"] = mean("sp_recall")
        out[f"{name}_all_gold"] = mean("all_gold")
        if not retrieval_only:
            out[f"{name}_em"] = mean("em")
            out[f"{name}_f1"] = mean("f1")
    out["cost_per_question_usd"] = round(sum(r["cost_usd"] for r in rows) / len(rows), 6)
    out["p50_ms"] = sorted(r["total_ms"] for r in rows)[len(rows) // 2]
    out["answer_model"] = "" if retrieval_only else config.LARGE_MODEL
    out["rerank_model"] = config.RERANK_MODEL if mode == "vector_rerank" or (
        mode.startswith("graph") and retrieve_mod.GRAPH_STRATEGY == "rerank") else ""
    out["graph_strategy"] = retrieve_mod.GRAPH_STRATEGY if mode.startswith("graph") else ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["vector", "vector_rerank", "graph", "graph_decompose"])
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--set", choices=["multi", "single"], default="multi",
                        help="multi: the HotpotQA sample; single: data/hotpotqa/single_hop.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    graph = linker = None
    if args.mode.startswith("graph"):
        from graphrag.graph import Neo4jGraph
        graph = Neo4jGraph()
        linker = graph.linker()
    if args.set == "multi":
        items = questions()
    else:
        items = json.loads((config.HOTPOT_DIR / "single_hop.json").read_text(encoding="utf-8"))
    items = items[:args.limit] if args.limit else items
    run_id = (f"{datetime.now():%Y%m%d-%H%M%S}-hotpot-{args.set}-{args.mode}"
              f"{'-retrieval' if args.retrieval_only else ''}")
    print(f"Evaluating {len(items)} questions, mode={args.mode}{' (retrieval only)' if args.retrieval_only else ''}")

    workers = 1 if args.retrieval_only else args.workers  # retrieval-only is local and CPU-bound
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda q: evaluate(q, args.mode, args.retrieval_only, graph, linker), items))
    qdrant().close()
    if graph:
        graph.close()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(run_id, args.mode, rows, args.retrieval_only)
    new = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="")
        if new:
            writer.writeheader()
        writer.writerow(summary)
    print("\n".join(f"{k:>24}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
