"""HTTP API around the pipeline.

Usage: uvicorn rag.api:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config, pipeline
from .store import embedder, qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder()  # load the model and open the store before the first request
    qdrant()
    yield
    qdrant().close()


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": qdrant().count(config.COLLECTION).count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = pipeline.answer(req.question, k=req.top_k)
    return AskResponse(
        answer=result.answer,
        sources=[Source(n=i, title=h.title, url=h.url, chunk_id=h.chunk_id, score=h.score, text=h.text)
                 for i, h in enumerate(result.sources, start=1)],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.total_ms,
    )
