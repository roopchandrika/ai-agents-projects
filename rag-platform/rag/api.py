"""HTTP API around the pipeline.

Usage: uvicorn rag.api:app --reload
"""
from contextlib import asynccontextmanager

from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import audit, config, secure, service
from .acl import User
from .auth import current_user, require_admin
from .store import embedder, qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder()  # load the model and open the store before the first request
    qdrant()
    yield
    qdrant().close()
    qdrant.cache_clear()


app = FastAPI(title="RAG baseline", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class Source(BaseModel):
    n: int
    title: str
    url: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool = False
    route: str | None = None
    escalated: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": qdrant().count(config.COLLECTION).count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return to_response(service.answer(req.question, k=req.top_k))


# ---- Multi-tenant service (Project 2) ----
# Identity comes only from the bearer token. The request models have no tenant or role fields, and any
# extra fields a client sends are ignored.

class TenantAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def to_response(served: service.Served) -> AskResponse:
    result = served.result
    return AskResponse(
        answer=result.answer,
        sources=[Source(n=i, title=h.title, url=h.url, chunk_id=h.chunk_id, score=h.score, text=h.text)
                 for i, h in enumerate(result.sources, start=1)],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.total_ms,
        cache_hit=served.cache_hit,
        route=served.route,
        escalated=served.escalated,
    )


@app.post("/tenant/ask", response_model=AskResponse)
def tenant_ask(req: TenantAskRequest, user: User = Depends(current_user)) -> AskResponse:
    return to_response(service.answer(req.question, user))


class AclUpdate(BaseModel):
    allowed_roles: list[str]
    allowed_users: list[str] = []
    classification: Literal["public", "internal", "confidential", "restricted"]


@app.put("/admin/documents/{doc_id:path}/acl")
def update_acl(doc_id: str, body: AclUpdate, admin: User = Depends(require_admin)) -> dict:
    try:
        result = secure.update_document_acl(admin, doc_id, body.allowed_roles, body.allowed_users,
                                            body.classification)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if result is None:
        raise HTTPException(404, "Document not found")
    return result


@app.get("/admin/documents/{doc_id:path}/access-log")
def access_log(doc_id: str, days: int = Query(30, ge=1, le=365), admin: User = Depends(require_admin)) -> dict:
    """Who retrieved this document in the last `days` days (within the admin's own tenant)."""
    return {"doc_id": doc_id, "days": days, "users": audit.who_retrieved(admin.tenant_id, doc_id, days)}


@app.get("/admin/alerts")
def security_alerts(days: int = Query(30, ge=1, le=365), admin: User = Depends(require_admin)) -> dict:
    return {"days": days, "alerts": audit.alerts(admin.tenant_id, days)}
