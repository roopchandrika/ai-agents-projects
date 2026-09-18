"""
retrieve.py
-----------
Query the Chroma collection built by ingest.py and format retrieved chunks
into a citation-tagged context block the chat loop can hand to Claude.
"""

import os
import pickle

import chromadb

from embeddings import Embedder


def load_collection(collection_name: str = "pdf_chunks", persist_dir: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    with open(os.path.join(persist_dir, "embedder.pkl"), "rb") as f:
        embedder: Embedder = pickle.load(f)
    return collection, embedder


def retrieve(query: str, collection, embedder: Embedder, top_k: int = 3) -> list[dict]:
    """Return the top_k most relevant chunks for `query`, each as
    {"text": ..., "source": ..., "page": ..., "distance": ...}."""
    query_vector = embedder.embed([query])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": distance,
        })
    return chunks


def format_citation(chunk: dict) -> str:
    return f"[{chunk['source']}, p.{chunk['page']}]"


def build_context(chunks: list[dict]) -> str:
    """Assemble retrieved chunks into a labeled context block for the
    prompt, each tagged with its citation so the model can cite its
    sources instead of inventing them."""
    blocks = []
    for chunk in chunks:
        blocks.append(f"{format_citation(chunk)}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)
