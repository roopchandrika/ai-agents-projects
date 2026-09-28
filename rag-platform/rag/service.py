"""The request path used by the API and the evals: semantic cache, then retrieval, routing and generation,
with every request written to the request log.

    cache hit?  -> return the stored answer (tenant scope: only if the user may still read every source)
    retrieve    -> public corpus, or permission-filtered tenant search (Project 2)
    route       -> small model for simple questions, large model for complex ones
    escalate    -> rerun on the large model with wider retrieval if the small model's answer looks weak
    store       -> cache answers that are complete and cited
"""
import os
import time
import uuid
from dataclasses import asdict, dataclass

from . import acl, audit, cache, config, pipeline, requestlog, router, secure
from .acl import User
from .pipeline import Answer, Hit
from .store import qdrant, store_lock

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
ROUTING_ENABLED = os.getenv("ROUTING_ENABLED", "1") == "1"


@dataclass
class Served:
    result: Answer
    request_id: str
    cache_hit: bool
    cache_similarity: float | None
    route: str | None
    route_reasons: list[str]
    escalated: bool
    escalation_reason: str | None


def _sources_still_allowed(sources: list[dict], user: User) -> bool:
    """Cached tenant answers are reusable only if this user may read every source chunk *right now*.
    Reads current ACLs from the tenant collection, so a revoke also invalidates cached answers."""
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, s["chunk_id"])) for s in sources]
    current = qdrant().retrieve(config.TENANT_COLLECTION, ids=ids, with_payload=True)
    return len(current) == len(ids) and all(acl.is_authorized(user, p.payload) for p in current)


def _cacheable(text: str, stop_reason: str, cited: bool) -> bool:
    return stop_reason == "end_turn" and bool(text.strip()) and cited and not router.is_unanswered(text)


def answer(question: str, user: User | None = None, *, use_cache: bool | None = None,
           use_routing: bool | None = None, run_id: str | None = None, k: int = config.TOP_K) -> Served:
    use_cache = CACHE_ENABLED if use_cache is None else use_cache
    use_routing = ROUTING_ENABLED if use_routing is None else use_routing
    request_id = str(uuid.uuid4())
    scope = user.tenant_id if user else "public"
    t0 = time.perf_counter()

    # 1. Semantic cache
    vector, verify_cost = None, 0.0
    if use_cache:
        vector = cache.key_vector(question)
        with store_lock:
            found = cache.lookup(vector, scope)
            if found and user is not None and not _sources_still_allowed(found[0]["sources"], user):
                found = None
        if found:
            same, verify_cost = cache.verify_same_answer(found[0]["query"], question)
            found = found if same else None
        if found:
            payload, similarity = found
            hits = [Hit(**s) for s in payload["sources"]]
            if user is not None:  # serving these documents' content is an access; audit it
                audit.log_retrieval(request_id, user, question, hits, "cache")
            elapsed = (time.perf_counter() - t0) * 1000
            result = Answer(question=question, answer=payload["answer"], sources=hits,
                            cited_doc_ids=pipeline.cited_doc_ids(payload["answer"], hits), model="cache",
                            stop_reason="cache_hit", input_tokens=0, output_tokens=0, cost_usd=verify_cost,
                            retrieval_ms=elapsed, generation_ms=0.0)
            served = Served(result, request_id, True, similarity, None, [], False, None)
            _log(served, scope, user, run_id)
            return served

    # 2. Retrieval
    with store_lock:
        t1 = time.perf_counter()
        hits = secure.secure_retrieve(question, user, k, request_id=request_id) if user else \
            pipeline.retrieve(question, k)
        t2 = time.perf_counter()

    # 3. Routing and generation
    decision = router.classify(question, hits) if use_routing else router.RouteDecision("complex", ["routing off"])
    model = config.SMALL_MODEL if decision.route == "simple" else config.LARGE_MODEL
    text, stop_reason, tokens_in, tokens_out = pipeline.generate(question, hits, model)
    cost = verify_cost + config.cost_usd(model, tokens_in, tokens_out)

    # 4. Escalation
    escalation = None
    if decision.route == "simple":
        escalation = router.needs_escalation(text, stop_reason, bool(pipeline.cited_doc_ids(text, hits)), hits)
        if escalation:  # larger model *and* wider retrieval
            model = config.LARGE_MODEL
            with store_lock:
                if user:
                    hits = secure.secure_retrieve(question, user, router.ESCALATION_TOP_K,
                                                  request_id=f"{request_id}:escalation")
                else:
                    hits = pipeline.retrieve(question, router.ESCALATION_TOP_K)
            text, stop_reason, large_in, large_out = pipeline.generate(question, hits, model)
            cost += config.cost_usd(model, large_in, large_out)
            tokens_in, tokens_out = tokens_in + large_in, tokens_out + large_out
    t3 = time.perf_counter()

    result = Answer(question=question, answer=text, sources=hits, cited_doc_ids=pipeline.cited_doc_ids(text, hits),
                    model=model, stop_reason=stop_reason, input_tokens=tokens_in, output_tokens=tokens_out,
                    cost_usd=cost, retrieval_ms=(t2 - t1) * 1000, generation_ms=(t3 - t2) * 1000)
    result.total_ms = (t3 - t0) * 1000  # include cache lookup time on a miss

    # 5. Store
    if use_cache and _cacheable(text, stop_reason, bool(result.cited_doc_ids)):
        with store_lock:
            cache.store(vector, scope, question, text, [asdict(h) for h in hits], model)

    served = Served(result, request_id, False, None, decision.route if use_routing else None,
                    decision.reasons, escalation is not None, escalation)
    _log(served, scope, user, run_id)
    return served


def _log(s: Served, scope: str, user: User | None, run_id: str | None) -> None:
    r = s.result
    requestlog.log(requestlog.RequestRecord(
        request_id=s.request_id, scope=scope, query=r.question, cache_hit=s.cache_hit,
        input_tokens=r.input_tokens, output_tokens=r.output_tokens, cost_usd=r.cost_usd, total_ms=r.total_ms,
        user_id=user.id if user else None, run_id=run_id, cache_similarity=s.cache_similarity, route=s.route,
        model=None if s.cache_hit else r.model, escalated=s.escalated, retrieval_ms=r.retrieval_ms,
        generation_ms=r.generation_ms, top_score=r.sources[0].score if r.sources else None,
        cited=bool(r.cited_doc_ids)))
