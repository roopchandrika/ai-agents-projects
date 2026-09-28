"""The plain RAG pipeline: retrieve top-k chunks, then generate a cited answer with Claude.

Every call returns timing, token and cost numbers so later projects have a baseline to beat.
"""
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import anthropic

from . import config
from .store import embed_query, qdrant

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided in the user's message.

Cite every factual claim with the number of the source it comes from, in square brackets, like [2] or [1][3].
If the sources don't contain the answer, say that you don't know based on the available documents instead of \
using outside knowledge. Keep answers concise: a few sentences unless the question needs more."""

# The embedded Qdrant store isn't built for concurrent access; serialize retrieval (it takes a few ms).
_retrieval_lock = threading.Lock()


@lru_cache(maxsize=1)
def claude() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit]
    cited_doc_ids: list[str]
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float = field(init=False)

    def __post_init__(self):
        self.total_ms = self.retrieval_ms + self.generation_ms

    def to_dict(self) -> dict:
        return asdict(self)


def retrieve(question: str, k: int = config.TOP_K) -> list[Hit]:
    result = qdrant().query_points(config.COLLECTION, query=embed_query(question), limit=k, with_payload=True)
    return [
        Hit(p.payload["chunk_id"], p.payload["doc_id"], p.payload["title"], p.payload["url"],
            p.payload["text"], p.score)
        for p in result.points
    ]


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, start=1))


def generate(question: str, hits: list[Hit], model: str = config.ANSWER_MODEL) -> tuple[str, str, int, int]:
    response = claude().messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<sources>\n{format_sources(hits)}\n</sources>\n\nQuestion: {question}",
        }],
    )
    if response.stop_reason == "refusal":
        text = "[refused]"
    else:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, response.stop_reason, response.usage.input_tokens, response.usage.output_tokens


def cited_doc_ids(answer: str, hits: list[Hit]) -> list[str]:
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted({hits[n - 1].doc_id for n in nums if 1 <= n <= len(hits)})


def answer(question: str, model: str = config.ANSWER_MODEL, k: int = config.TOP_K) -> Answer:
    with _retrieval_lock:
        t0 = time.perf_counter()
        hits = retrieve(question, k)
        t1 = time.perf_counter()
    text, stop_reason, tokens_in, tokens_out = generate(question, hits, model)
    t2 = time.perf_counter()
    return Answer(
        question=question, answer=text, sources=hits, cited_doc_ids=cited_doc_ids(text, hits),
        model=model, stop_reason=stop_reason, input_tokens=tokens_in, output_tokens=tokens_out,
        cost_usd=config.cost_usd(model, tokens_in, tokens_out),
        retrieval_ms=(t1 - t0) * 1000, generation_ms=(t2 - t1) * 1000,
    )
