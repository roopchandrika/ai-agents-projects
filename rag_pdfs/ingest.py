"""
ingest.py
---------
Extract text from PDFs page-by-page, split into overlapping chunks, embed
them, and store them (plus source/page metadata) in a local Chroma
collection so they can be retrieved later with citations.
"""

import os
import uuid

import chromadb
from pypdf import PdfReader

from embeddings import Embedder, TfidfEmbedder

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150


def extract_pages(pdf_path: str) -> list[str]:
    """Return a list of page texts, indexed by page number (0-based)."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    Splits on whitespace boundaries where possible so chunks don't cut a
    word in half. `overlap` characters are repeated between consecutive
    chunks so a sentence spanning a chunk boundary isn't lost from context.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # back up to the nearest preceding space so we don't split a word
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_corpus(pdf_paths: list[str]) -> list[dict]:
    """Extract + chunk every PDF into a flat list of
    {"text": ..., "source": filename, "page": page_number} records."""
    records = []
    for pdf_path in pdf_paths:
        source = os.path.basename(pdf_path)
        pages = extract_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            for chunk in chunk_text(page_text):
                records.append({"text": chunk, "source": source, "page": page_num})
    return records


def ingest(
    pdf_paths: list[str],
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./chroma_db",
    embedder: Embedder | None = None,
) -> chromadb.Collection:
    """Build the corpus from the given PDFs, embed it, and store it in a
    persistent Chroma collection. Returns (collection, embedder).

    The fitted embedder is also pickled into `persist_dir` because TF-IDF
    vectors are only meaningful relative to the vocabulary they were fit
    on -- a query at chat time must be embedded with this SAME fitted
    embedder, not a freshly-fit one.
    """
    import pickle

    embedder = embedder or TfidfEmbedder()

    records = build_corpus(pdf_paths)
    if not records:
        raise ValueError("No extractable text found in the given PDFs.")

    texts = [r["text"] for r in records]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    os.makedirs(persist_dir, exist_ok=True)
    with open(os.path.join(persist_dir, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)

    client = chromadb.PersistentClient(path=persist_dir)
    # Start clean each ingest run so re-running doesn't duplicate chunks.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    collection.add(
        ids=[str(uuid.uuid4()) for _ in records],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
    )
    return collection, embedder


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf1> [pdf2 ...]")
        sys.exit(1)
    collection, _ = ingest(sys.argv[1:])
    print(f"Ingested {collection.count()} chunks from {len(sys.argv) - 1} PDF(s) into ./chroma_db")
