"""Measure what access control costs: search latency with and without the ACL filter, and how long a
permission revoke takes to apply.

Usage: python -m scripts.measure_acl [--passes 5] [--revokes 20]
Needs the tenant collection with filler (python -m rag.tenant_ingest). Writes eval/results/acl_metrics.json.
Revokes go through the real admin path, so they appear in the audit log's acl_changes table.
"""
import argparse
import json
import statistics
import time

from eval.leakage_cases import load_manifest, users
from rag import config, secure
from rag.acl import acl_filter
from rag.store import embed_query, qdrant


def pct(values: list[float], p: int) -> float:
    return round(statistics.quantiles(values, n=100, method="inclusive")[p - 1], 3)


def search_ms(vector, query_filter) -> float:
    t0 = time.perf_counter()
    qdrant().query_points(config.TENANT_COLLECTION, query=vector, query_filter=query_filter,
                          limit=config.TOP_K, with_payload=True)
    return (time.perf_counter() - t0) * 1000


def filter_overhead(manifest: dict, passes: int) -> dict:
    people = users(manifest)
    questions = [json.loads(line)["question"] for line in
                 (config.EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    questions += [d["title"] for d in manifest["documents"]]
    vectors = [embed_query(q) for q in questions]  # embedding time is the same either way; leave it out
    for v in vectors[:10]:  # warm-up
        search_ms(v, None)

    plain, filtered = [], []
    for _ in range(passes):
        for i, v in enumerate(vectors):
            plain.append(search_ms(v, None))
            filtered.append(search_ms(v, acl_filter(people[i % len(people)])))
    return {
        "searches_per_mode": len(plain),
        "unfiltered_p50_ms": pct(plain, 50), "unfiltered_p95_ms": pct(plain, 95),
        "filtered_p50_ms": pct(filtered, 50), "filtered_p95_ms": pct(filtered, 95),
        "overhead_p50_ms": round(pct(filtered, 50) - pct(plain, 50), 3),
        "overhead_p95_ms": round(pct(filtered, 95) - pct(plain, 95), 3),
    }


def revoke_latency(manifest: dict, trials: int) -> dict:
    """Time from the admin's revoke call until the next filtered search no longer returns the document."""
    people = {u.id: u for u in users(manifest)}
    doc = next(d for d in manifest["documents"] if d["allowed_users"])
    reader = people[doc["allowed_users"][0]]
    admin = next(u for u in people.values() if u.tenant_id == doc["tenant_id"] and "admin" in u.roles)
    vector = embed_query(doc["title"])

    def visible() -> bool:
        points = qdrant().query_points(config.TENANT_COLLECTION, query=vector, query_filter=acl_filter(reader),
                                       limit=20, with_payload=True).points
        return any(p.payload["doc_id"] == doc["doc_id"] for p in points)

    timings, stale_reads = [], 0
    for _ in range(trials):
        assert visible(), "document should be visible before the revoke"
        t0 = time.perf_counter()
        secure.update_document_acl(admin, doc["doc_id"], doc["allowed_roles"], [], doc["classification"])
        stale_reads += visible()  # the very next query after the call returns
        timings.append((time.perf_counter() - t0) * 1000)
        secure.update_document_acl(admin, doc["doc_id"], doc["allowed_roles"], doc["allowed_users"],
                                   doc["classification"])
    return {"document": doc["doc_id"], "trials": trials, "stale_reads_after_revoke": stale_reads,
            "revoke_to_enforced_p50_ms": pct(timings, 50), "revoke_to_enforced_max_ms": round(max(timings), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--revokes", type=int, default=20)
    args = parser.parse_args()

    manifest = load_manifest()
    chunks = qdrant().count(config.TENANT_COLLECTION).count
    results = {
        "collection_chunks": chunks,
        "qdrant_mode": "server" if config.QDRANT_URL else "embedded (local, no payload indexes)",
        "filter": filter_overhead(manifest, args.passes),
        "revoke": revoke_latency(manifest, args.revokes),
    }
    qdrant().close()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "acl_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
