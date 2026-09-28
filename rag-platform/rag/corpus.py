"""Loading documents and splitting them into overlapping token-window chunks."""
import hashlib
import json
from dataclasses import dataclass

from . import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    index: int
    text: str


def load_documents() -> list[dict]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.CORPUS_DIR.glob("*.json"))]
    if not docs:
        raise SystemExit(f"No documents in {config.CORPUS_DIR}. Run: python -m scripts.fetch_corpus")
    return docs


def corpus_version(docs: list[dict]) -> str:
    """Short hash of the corpus contents. Changes whenever any document changes."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d["doc_id"]):
        h.update(d["doc_id"].encode())
        h.update(d["text"].encode())
    return h.hexdigest()[:12]


def chunk_document(doc: dict, tokenizer, size: int = config.CHUNK_TOKENS,
                   overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Slide a window of `size` tokens with `overlap` tokens of overlap, cutting the
    original text at token boundaries so chunks keep their exact wording."""
    text = doc["text"]
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    chunks, start, step = [], 0, size - overlap
    while start < len(offsets):
        end = min(start + size, len(offsets))
        piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        i = len(chunks)
        chunks.append(Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], doc["title"], doc["url"], i, piece))
        if end == len(offsets):
            break
        start += step
    return chunks


def chunk_all(docs: list[dict], tokenizer) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, tokenizer)]
