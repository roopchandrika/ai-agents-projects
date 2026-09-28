"""Chunk the corpus, embed every chunk and (re)build the Qdrant collection.

Usage: python -m rag.ingest
"""
import uuid

from qdrant_client import models

from . import config
from .corpus import chunk_all, corpus_version, load_documents
from .store import embed_passages, embedder, passage_text, qdrant


def main() -> None:
    docs = load_documents()
    version = corpus_version(docs)
    chunks = chunk_all(docs, embedder().tokenizer)
    print(f"{len(docs)} documents -> {len(chunks)} chunks (corpus version {version})")

    vectors = embed_passages([passage_text(c.title, c.text) for c in chunks])

    client = qdrant()
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector=v,
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "url": c.url,
                     "chunk_index": c.index, "text": c.text, "corpus_version": version},
        )
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(config.COLLECTION, points=points[i:i + 256])
    print(f"Indexed {len(points)} chunks into '{config.COLLECTION}'")
    client.close()


if __name__ == "__main__":
    main()
