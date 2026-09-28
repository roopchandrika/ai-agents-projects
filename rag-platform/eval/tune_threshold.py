"""Tune the semantic cache on the paraphrase / near-miss pairs.

Stage 1 sweeps the embedding-similarity threshold. A paraphrase above it is a correct hit; a near-miss above it
would be a wrong answer served from cache. On this corpus no threshold separates the two, so the threshold is used
only to nominate candidates: the highest value that still catches TARGET_RECALL of paraphrases.
Stage 2 (--verify) runs the verifier model on every pair above that threshold and reports end-to-end numbers.

Usage: python -m eval.tune_threshold [--verify]
       -> eval/results/threshold_sweep.csv, eval/results/cache_tuning.json
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rag import cache, config

PAIRS_PATH = config.EVAL_DIR / "cache_pairs.jsonl"
TARGET_RECALL = 0.97


def sweep(pos: np.ndarray, neg: np.ndarray) -> list[dict]:
    rows = []
    for t in np.round(np.arange(0.80, 0.9951, 0.005), 3):
        rows.append({"threshold": float(t), "paraphrase_hit_rate": round(float((pos >= t).mean()), 3),
                     "false_hits": int((neg >= t).sum()), "false_hit_rate": round(float((neg >= t).mean()), 3)})
    return rows


def verify(pairs: list[dict], pos: np.ndarray, neg: np.ndarray, candidate: float) -> dict:
    jobs = [(x["question"], x["paraphrase"], True) for x, s in zip(pairs, pos) if s >= candidate]
    jobs += [(x["question"], x["near_miss"], False) for x, s in zip(pairs, neg) if s >= candidate]
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(lambda j: cache.verify_same_answer(j[0], j[1]), jobs))
    served = sum(1 for j, (same, _) in zip(jobs, verdicts) if j[2] and same)
    false_hits = [(j[0], j[1]) for j, (same, _) in zip(jobs, verdicts) if not j[2] and same]
    print(f"\nWith {config.CACHE_VERIFY_MODEL} verifying candidates: {served}/{len(pairs)} paraphrases served, "
          f"{len(false_hits)} false hits out of {sum(1 for j in jobs if not j[2])} near-misses that reached it")
    for a, b in false_hits:
        print(f"  FALSE HIT: {a[:70]}\n             {b[:70]}")
    return {
        "verify_model": config.CACHE_VERIFY_MODEL,
        "end_to_end_paraphrase_recall": round(served / len(pairs), 3),
        "end_to_end_false_hits": len(false_hits),
        "near_misses_reaching_verifier": sum(1 for j in jobs if not j[2]),
        "verify_cost_per_check_usd": round(sum(c for _, c in verdicts) / len(verdicts), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run the verifier model on candidates (~$0.05)")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    vec = lambda texts: np.array([cache.key_vector(t) for t in texts])
    q, p, n = (vec([x[k] for x in pairs]) for k in ("question", "paraphrase", "near_miss"))
    pos, neg = (q * p).sum(1), (q * n).sum(1)

    rows = sweep(pos, neg)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(pairs)} pairs. Paraphrase similarity: median {np.median(pos):.3f}, min {pos.min():.3f}. "
          f"Near-miss similarity: median {np.median(neg):.3f}, max {neg.max():.3f}\n")
    print(f"{'threshold':>9} {'para hits':>9} {'false hits':>10}")
    for r in rows:
        print(f"{r['threshold']:>9.3f} {r['paraphrase_hit_rate']:>9.0%} {r['false_hits']:>10}")

    safe = [r for r in rows if r["false_hits"] == 0]
    candidate = max(r["threshold"] for r in rows if r["paraphrase_hit_rate"] >= TARGET_RECALL)
    print(f"\nLowest threshold with zero false hits: {safe[0]['threshold']} "
          f"(catches {safe[0]['paraphrase_hit_rate']:.0%} of paraphrases)")
    print(f"Candidate threshold (>= {TARGET_RECALL:.0%} paraphrase recall): {candidate}")
    print("\nHardest near-misses:")
    for s, x in sorted(zip(neg, pairs), key=lambda t: -t[0])[:5]:
        print(f"  {s:.3f}  {x['question'][:70]}\n         {x['near_miss'][:70]}")

    summary = {"pairs": len(pairs), "paraphrase_similarity_median": round(float(np.median(pos)), 3),
               "near_miss_similarity_median": round(float(np.median(neg)), 3),
               "zero_false_hit_threshold": safe[0]["threshold"],
               "zero_false_hit_paraphrase_recall": safe[0]["paraphrase_hit_rate"],
               "candidate_threshold": candidate}
    if args.verify:
        summary.update(verify(pairs, pos, neg, candidate))
    (config.RESULTS_DIR / "cache_tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
