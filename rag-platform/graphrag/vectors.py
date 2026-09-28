"""Paragraph embeddings in Qdrant (the same store and embedding model as the rest of the platform).

Usage: python -m graphrag.vectors     # (re)index the HotpotQA paragraphs
"""
import uuid

from qdrant_client import models

from rag import config
from rag.store import embed_passages, embed_query, qdrant, reset_collection, store_lock

from .corpus import paragraphs


def index() -> int:
    paras = list(paragraphs().values())
    vectors = embed_passages([f"{p.title}\n\n{p.text}" for p in paras])
    reset_collection(config.HOTPOT_COLLECTION, models.VectorParams(size=len(vectors[0]),
                                                                   distance=models.Distance.COSINE))
    points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"hotpot/{p.title}")), vector=v,
                                 payload={"title": p.title}) for p, v in zip(paras, vectors)]
    for i in range(0, len(points), 256):
        qdrant().upsert(config.HOTPOT_COLLECTION, points=points[i:i + 256])
    return len(points)


def search(question: str, k: int) -> list[tuple[str, float]]:
    """(title, similarity) for the k nearest paragraphs."""
    vector = embed_query(question)
    with store_lock:
        points = qdrant().query_points(config.HOTPOT_COLLECTION, query=vector, limit=k,
                                       with_payload=True).points
    return [(p.payload["title"], p.score) for p in points]


def similarity(question: str, titles: list[str]) -> dict[str, float]:
    """Cosine similarity between the question and specific paragraphs, from their stored vectors."""
    if not titles:
        return {}
    q = embed_query(question)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"hotpot/{t}")) for t in titles]
    with store_lock:
        points = qdrant().retrieve(config.HOTPOT_COLLECTION, ids=ids, with_payload=True, with_vectors=True)
    return {p.payload["title"]: sum(a * b for a, b in zip(q, p.vector)) for p in points}


if __name__ == "__main__":
    print(f"Indexed {index()} paragraphs into '{config.HOTPOT_COLLECTION}'")
    qdrant().close()
