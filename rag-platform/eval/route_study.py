"""Label questions by what actually happens, then compare routing policies offline.

For every question (the 78 single-fact eval questions plus the 28 two-article questions) this runs the small
model, the large model (reusing the baseline run's large-model answers where they exist) and the LLM judge, and
records cost, the escalation signal and both routers' decisions. A question is labelled "complex" when the large
model gets it right and the small one doesn't.

Because both answers are known, any policy can be simulated without re-running it: always-large, always-small,
rules, rules + escalation, small + escalation, LLM classifier + escalation.

Usage: python -m eval.route_study --baseline eval/results/<baseline>.csv
       -> eval/results/route_study.jsonl, eval/results/route_study.json
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from eval.run_eval import judge
from rag import config, pipeline, router
from rag.pipeline import claude
from rag.store import store_lock

CLASSIFIER_PROMPT = """A retrieval-augmented Q&A system answers space-exploration questions from five retrieved passages. \
It can use a small, fast model or a larger, more capable one. Decide which this question needs.

<question>{question}</question>

"simple": a single fact or short description that can be read straight out of one passage.
"complex": comparing things, explaining why or how, combining facts from several passages, or several sub-questions."""


class RouteLabel(BaseModel):
    route: str


def classify_with_llm(question: str) -> tuple[str, float]:
    r = claude().messages.parse(model=config.SMALL_MODEL, max_tokens=64, output_format=RouteLabel,
                                messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question)}])
    route = "complex" if "complex" in r.parsed_output.route.lower() else "simple"
    return route, config.cost_usd(config.SMALL_MODEL, r.usage.input_tokens, r.usage.output_tokens)


def run(model: str, question: str, hits) -> dict:
    text, stop, tin, tout = pipeline.generate(question, hits, model)
    return {"answer": text, "stop_reason": stop, "cost": config.cost_usd(model, tin, tout),
            "cited": bool(pipeline.cited_doc_ids(text, hits))}


def study(item: dict, baseline: dict | None) -> dict:
    with store_lock:
        hits = pipeline.retrieve(item["question"])
    small = run(config.SMALL_MODEL, item["question"], hits)
    small["correct"] = judge(item["question"], item["reference_answer"], small["answer"])[0].correct
    if baseline:
        large = {"answer": baseline["answer"], "cost": float(baseline["cost_usd"]), "correct": baseline["correct"] == "1"}
    else:
        large = run(config.LARGE_MODEL, item["question"], hits)
        large["correct"] = judge(item["question"], item["reference_answer"], large["answer"])[0].correct
    llm_route, llm_cost = classify_with_llm(item["question"])
    decision = router.classify(item["question"], hits)
    return {
        "id": item["id"], "set": "complex" if item["id"].startswith("c") else "single-fact",
        "question": item["question"], "small_correct": small["correct"], "large_correct": large["correct"],
        "small_cost": small["cost"], "large_cost": large["cost"],
        "escalation": router.needs_escalation(small["answer"], small["stop_reason"], small["cited"], hits),
        "rule_route": decision.route, "rule_reasons": decision.reasons, "llm_route": llm_route,
        "llm_classifier_cost": llm_cost, "features": router.features(item["question"], hits),
        "label": "complex" if large["correct"] and not small["correct"] else "simple",
    }


def simulate(rows: list[dict]) -> dict:
    """Accuracy and mean cost per question for each policy, from the recorded answers."""
    def outcome(row, route, escalate, extra_cost=0.0):
        if route == "complex":
            return row["large_correct"], row["large_cost"] + extra_cost
        if escalate and row["escalation"]:
            return row["large_correct"], row["small_cost"] + row["large_cost"] + extra_cost
        return row["small_correct"], row["small_cost"] + extra_cost

    policies = {
        "always large (baseline)": lambda r: outcome(r, "complex", False),
        "always small": lambda r: outcome(r, "simple", False),
        "small + escalation": lambda r: outcome(r, "simple", True),
        "rules": lambda r: outcome(r, r["rule_route"], False),
        "rules + escalation": lambda r: outcome(r, r["rule_route"], True),
        "LLM classifier + escalation": lambda r: outcome(r, r["llm_route"], True, r["llm_classifier_cost"]),
    }
    out = {}
    for name, fn in policies.items():
        results = [fn(r) for r in rows]
        out[name] = {"accuracy": round(sum(c for c, _ in results) / len(rows), 4),
                     "cost_per_question_usd": round(sum(c for _, c in results) / len(rows), 6),
                     "large_model_share": round(sum(1 for r in rows if (r["rule_route"] if "rules" in name else
                                                   r["llm_route"] if "LLM" in name else
                                                   "complex" if "large" in name else "simple") == "complex"
                                                   or ("escalation" in name and r["escalation"])) / len(rows), 3)}
    return out


def router_accuracy(rows: list[dict], key: str) -> dict:
    decisive = [r for r in rows if r["small_correct"] != r["large_correct"]]
    return {"agreement_with_labels": round(sum(r[key] == r["label"] for r in rows) / len(rows), 3),
            "complex_questions_caught": f"{sum(r[key] == 'complex' for r in rows if r['label'] == 'complex')}"
                                        f"/{sum(r['label'] == 'complex' for r in rows)}",
            "decisive_questions": len(decisive)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="baseline run CSV with large-model answers")
    args = parser.parse_args()

    baseline = {r["id"]: r for r in csv.DictReader(open(args.baseline, encoding="utf-8"))}
    items = []
    for name in ("questions.jsonl", "questions_complex.jsonl"):
        items += [json.loads(l) for l in (config.EVAL_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Studying {len(items)} questions...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda it: study(it, baseline.get(it["id"])), items))

    with (config.RESULTS_DIR / "route_study.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"questions": len(rows),
               "labels": {k: sum(r["label"] == k for r in rows) for k in ("simple", "complex")},
               "small_accuracy_by_set": {s: round(sum(r["small_correct"] for r in rows if r["set"] == s) /
                                                  sum(r["set"] == s for r in rows), 3) for s in ("single-fact", "complex")},
               "large_accuracy_by_set": {s: round(sum(r["large_correct"] for r in rows if r["set"] == s) /
                                                  sum(r["set"] == s for r in rows), 3) for s in ("single-fact", "complex")},
               "rules": router_accuracy(rows, "rule_route"), "llm_classifier": router_accuracy(rows, "llm_route"),
               "policies": simulate(rows)}
    (config.RESULTS_DIR / "route_study.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
