"""The leakage suite: adversarial queries from every persona must never retrieve a forbidden chunk."""
from collections import defaultdict

import pytest

from eval.leakage_cases import allowed_doc_ids, cases, is_allowed_doc, load_manifest
from rag import config
from rag.secure import secure_retrieve
from rag.store import embed_query, qdrant

K = 10  # deeper than production top 5, to catch leaks that would rank lower
CASES_BY_USER = defaultdict(list)
for case in cases():
    CASES_BY_USER[case.user.id].append(case)


def test_suite_is_large_enough():
    total = sum(len(v) for v in CASES_BY_USER.values())
    print(f"\n{total} adversarial queries across {len(CASES_BY_USER)} personas")
    assert total >= 500


@pytest.mark.parametrize("user_id", sorted(CASES_BY_USER))
def test_no_forbidden_chunks_retrieved(user_id, manifest):
    violations = []
    for case in CASES_BY_USER[user_id]:
        for hit in secure_retrieve(case.query, case.user, k=K):
            if not is_allowed_doc(case.user, hit.doc_id, manifest):
                violations.append((case.query, hit.chunk_id))
    assert violations == []


def test_attacks_would_succeed_without_the_filter(manifest):
    """Proves the suite has teeth: with no ACL filter, the targeted queries do pull up their forbidden docs."""
    targeted = [c for c in cases(manifest) if c.target_doc]
    found = 0
    for case in targeted:
        points = qdrant().query_points(config.TENANT_COLLECTION, query=embed_query(case.query), limit=K,
                                       with_payload=True).points
        found += any(p.payload["doc_id"] == case.target_doc for p in points)
    rate = found / len(targeted)
    print(f"\nWithout the filter, {rate:.0%} of targeted attacks retrieve their forbidden document")
    assert rate > 0.5


@pytest.mark.parametrize("user_id", sorted(CASES_BY_USER))
def test_authorized_documents_stay_reachable(user_id, users, manifest):
    """Guards against a filter that is 'secure' because it returns nothing."""
    user = users[user_id]
    titles = {d["doc_id"]: d["title"] for d in manifest["documents"]}
    missing = [doc_id for doc_id in allowed_doc_ids(user, manifest)
               if doc_id not in {h.doc_id for h in secure_retrieve(titles[doc_id], user, k=K)}]
    assert missing == []
