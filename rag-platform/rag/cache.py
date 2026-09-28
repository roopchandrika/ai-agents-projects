"""Semantic cache: answers keyed by the meaning of the question, stored in their own Qdrant collection.

An entry is only reused when all of these hold:
- the new question's similarity to the cached one is at or above the tuned threshold,
- it was cached in the same scope (the public corpus, or one tenant),
- it was built from the current corpus version (re-ingesting makes old entries unreachable),
- it hasn't expired (TTL, a second safety net).
- the verifier model confirms the two questions have the same answer (skipped for identical normalized text).
Tenant entries are additionally re-checked against the asking user's current permissions (see service.py).

Why the verifier: on this corpus, near-misses ("Pioneer 10" vs "Pioneer 11", "June 30" vs "September 30") score
*higher* embedding similarity than true paraphrases, so no similarity threshold alone gives zero wrong answers.
"""
import hashlib
import re
import time
import uuid
from functools import lru_cache

from pydantic import BaseModel
from qdrant_client import models

from . import config
from .pipeline import claude
from .store import embedder, qdrant, reset_collection

_PUNCT = re.compile(r"[^\w\s]")


def normalize(query: str) -> str:
    return " ".join(_PUNCT.sub(" ", query.lower()).split())


def key_vector(query: str) -> list[float]:
    """Question-to-question similarity, so no retrieval prefix here (unlike embed_query)."""
    return embedder().encode(normalize(query), normalize_embeddings=True).tolist()


@lru_cache(maxsize=None)
def corpus_version(scope: str) -> str:
    """Version tag of the data a cached answer depends on. Public: the baseline corpus hash stored at ingest.
    Tenant: a hash of the manifest and that tenant's documents."""
    if scope == "public":
        if not qdrant().collection_exists(config.COLLECTION):
            return "empty"
        points, _ = qdrant().scroll(config.COLLECTION, limit=1, with_payload=["corpus_version"])
        return points[0].payload["corpus_version"] if points else "empty"
    h = hashlib.sha256((config.TENANT_DIR / "manifest.json").read_bytes())
    for path in sorted((config.TENANT_DIR / scope).glob("*.md")):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _ensure_collection() -> None:
    client = qdrant()
    if not client.collection_exists(config.CACHE_COLLECTION):
        dim = embedder().get_embedding_dimension()
        client.create_collection(config.CACHE_COLLECTION, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE))
        for field in ("scope", "corpus_version"):
            client.create_payload_index(config.CACHE_COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(config.CACHE_COLLECTION, "expires_at", models.PayloadSchemaType.FLOAT)


def lookup(vector: list[float], scope: str, threshold: float | None = None) -> tuple[dict, float] | None:
    """Nearest live entry in this scope. Returns (payload, similarity) at or above the threshold, else None."""
    threshold = config.CACHE_THRESHOLD if threshold is None else threshold
    _ensure_collection()
    live = models.Filter(must=[
        models.FieldCondition(key="scope", match=models.MatchValue(value=scope)),
        models.FieldCondition(key="corpus_version", match=models.MatchValue(value=corpus_version(scope))),
        models.FieldCondition(key="expires_at", range=models.Range(gt=time.time())),
    ])
    points = qdrant().query_points(config.CACHE_COLLECTION, query=vector, query_filter=live, limit=1,
                                   with_payload=True).points
    if points and points[0].score >= threshold:
        return points[0].payload, points[0].score
    return None


def store(vector: list[float], scope: str, query: str, answer: str, sources: list[dict], model: str) -> None:
    _ensure_collection()
    now = time.time()
    qdrant().upsert(config.CACHE_COLLECTION, points=[models.PointStruct(
        id=str(uuid.uuid4()), vector=vector,
        payload={"scope": scope, "query": query, "answer": answer, "sources": sources, "model": model,
                 "corpus_version": corpus_version(scope), "created_at": now,
                 "expires_at": now + config.CACHE_TTL_HOURS * 3600})])


def clear() -> None:
    reset_collection(config.CACHE_COLLECTION)
    corpus_version.cache_clear()


VERIFY_PROMPT = """A cache returns a stored answer to question A when a user asks question B. Decide whether that would be correct.

<question_a>{a}</question_a>
<question_b>{b}</question_b>

Answer "same" only if a complete, correct answer to A is also a complete, correct answer to B: same entity, same attribute or event, same qualifiers (dates, numbers, first/last, which part of a multi-part question). Different wording is fine. If B asks for anything A's answer would not contain, answer "different"."""


class _Verdict(BaseModel):
    verdict: str


def verify_same_answer(cached_query: str, new_query: str) -> tuple[bool, float]:
    """(same_answer, cost_usd). Identical normalized questions skip the model call."""
    if normalize(cached_query) == normalize(new_query):
        return True, 0.0
    r = claude().messages.parse(
        model=config.CACHE_VERIFY_MODEL, max_tokens=64, output_format=_Verdict,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(a=cached_query, b=new_query)}])
    same = r.stop_reason == "end_turn" and r.parsed_output.verdict.strip().lower() == "same"
    return same, config.cost_usd(config.CACHE_VERIFY_MODEL, r.usage.input_tokens, r.usage.output_tokens)
