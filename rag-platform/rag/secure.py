"""Permission-aware retrieval and answering for the multi-tenant service.

1. The ACL filter is built from the verified user and applied inside the vector search (pre-filtering),
   so forbidden chunks never compete for the top k.
2. Every returned chunk is checked again in application code; anything that fails is dropped and alerted.
3. Every retrieval is written to the append-only audit log before anything reaches the LLM.
"""
import logging
import time
import uuid

from qdrant_client import models

from . import acl, audit, config, pipeline
from .acl import User
from .pipeline import Answer, Hit
from .store import embed_query, qdrant, store_lock

log = logging.getLogger("rag.secure")


def secure_retrieve(question: str, user: User, k: int = config.TOP_K, request_id: str | None = None,
                    model: str | None = None) -> list[Hit]:
    request_id = request_id or str(uuid.uuid4())
    result = qdrant().query_points(config.TENANT_COLLECTION, query=embed_query(question),
                                   query_filter=acl.acl_filter(user), limit=k, with_payload=True)
    hits = []
    for p in result.points:
        if not acl.is_authorized(user, p.payload):
            log.error("ACL filter returned forbidden chunk %s to %s", p.payload.get("chunk_id"), user.id)
            audit.log_alert(request_id, user, "forbidden chunk returned by vector search",
                            p.payload.get("chunk_id"), p.payload.get("doc_id"))
            continue
        hits.append(Hit(p.payload["chunk_id"], p.payload["doc_id"], p.payload["title"], p.payload["url"],
                        p.payload["text"], p.score))
    audit.log_retrieval(request_id, user, question, hits, model)
    return hits


def secure_answer(question: str, user: User, model: str = config.ANSWER_MODEL, k: int = config.TOP_K) -> Answer:
    with store_lock:
        t0 = time.perf_counter()
        hits = secure_retrieve(question, user, k, model=model)
        t1 = time.perf_counter()
    text, stop_reason, tokens_in, tokens_out = pipeline.generate(question, hits, model)
    t2 = time.perf_counter()
    return Answer(
        question=question, answer=text, sources=hits, cited_doc_ids=pipeline.cited_doc_ids(text, hits),
        model=model, stop_reason=stop_reason, input_tokens=tokens_in, output_tokens=tokens_out,
        cost_usd=config.cost_usd(model, tokens_in, tokens_out),
        retrieval_ms=(t1 - t0) * 1000, generation_ms=(t2 - t1) * 1000,
    )


def update_document_acl(admin: User, doc_id: str, allowed_roles: list[str], allowed_users: list[str],
                        classification: str) -> dict | None:
    """Change the ACL on every chunk of a document in the admin's own tenant. Returns None if the
    document doesn't exist there (other tenants' documents are indistinguishable from missing ones)."""
    scope = models.Filter(must=[
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=admin.tenant_id)),
    ])
    new = {"tenant_id": admin.tenant_id, "allowed_roles": allowed_roles, "allowed_users": allowed_users,
           "classification": classification}
    acl.validate_acl({"chunk_id": doc_id, **new})
    with store_lock:
        existing, _ = qdrant().scroll(config.TENANT_COLLECTION, scroll_filter=scope, limit=1, with_payload=True)
        if not existing:
            return None
        old = {f: existing[0].payload.get(f) for f in acl.ACL_FIELDS}
        t0 = time.perf_counter()
        qdrant().set_payload(config.TENANT_COLLECTION, payload=new, points=scope, wait=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    audit.log_acl_change(admin, doc_id, old, new)
    return {"doc_id": doc_id, "old": old, "new": new, "update_ms": round(elapsed_ms, 2)}
