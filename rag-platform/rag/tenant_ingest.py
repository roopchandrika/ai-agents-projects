"""Build the multi-tenant collection: seed company documents with their ACLs, plus the Wikipedia corpus
split across tenants as filler so the index is a realistic size.

Usage: python -m rag.tenant_ingest [--no-filler]

Every chunk inherits its document's ACL and is validated before it is written; a chunk without a
tenant_id stops ingestion. Filler reuses the vectors from the baseline collection instead of re-embedding.
"""
import argparse
import hashlib
import json
import uuid

from qdrant_client import models

from . import config
from .acl import validate_acl
from .corpus import chunk_document
from .store import embed_passages, embedder, passage_text, qdrant

INDEXED_FIELDS = ("tenant_id", "allowed_roles", "allowed_users", "doc_id")


def load_manifest() -> dict:
    return json.loads((config.TENANT_DIR / "manifest.json").read_text(encoding="utf-8"))


def acl_of(doc: dict) -> dict:
    return {"tenant_id": doc["tenant_id"], "allowed_roles": list(doc["allowed_roles"]),
            "allowed_users": list(doc["allowed_users"]), "classification": doc["classification"]}


def point(chunk_id: str, vector: list[float], payload: dict) -> models.PointStruct:
    validate_acl(payload)  # raises: an unprotected chunk is never written
    return models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)), vector=vector, payload=payload)


def seed_points(manifest: dict) -> list[models.PointStruct]:
    tokenizer = embedder().tokenizer
    chunks, acls = [], []
    for doc in manifest["documents"]:
        text = (config.TENANT_DIR / doc["path"]).read_text(encoding="utf-8")
        record = {"doc_id": doc["doc_id"], "title": doc["title"], "url": f"tenant://{doc['doc_id']}", "text": text}
        for c in chunk_document(record, tokenizer):
            chunks.append(c)
            acls.append(acl_of(doc))
    vectors = embed_passages([passage_text(c.title, c.text) for c in chunks])
    return [point(c.chunk_id, v, {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "url": c.url,
                                  "text": c.text, **acl})
            for c, v, acl in zip(chunks, vectors, acls)]


def filler_points(tenants: list[str]) -> list[models.PointStruct]:
    """Each Wikipedia article goes to one tenant (stable hash), readable by all its employees."""
    client = qdrant()
    if not client.collection_exists(config.COLLECTION):
        print(f"Baseline collection '{config.COLLECTION}' not found; skipping filler (run python -m rag.ingest)")
        return []
    points, offset = [], None
    while True:
        batch, offset = client.scroll(config.COLLECTION, limit=512, offset=offset,
                                      with_payload=True, with_vectors=True)
        for p in batch:
            src = p.payload
            tenant = tenants[int(hashlib.sha256(src["doc_id"].encode()).hexdigest(), 16) % len(tenants)]
            doc_id = f"{tenant}/wiki-{src['doc_id']}"
            chunk_id = f"{doc_id}#{src['chunk_index']}"
            points.append(point(chunk_id, p.vector, {
                "chunk_id": chunk_id, "doc_id": doc_id, "title": src["title"], "url": src["url"],
                "text": src["text"], "tenant_id": tenant, "allowed_roles": ["employee"], "allowed_users": [],
                "classification": "public"}))
        if offset is None:
            return points


def ingest(include_filler: bool = True) -> int:
    manifest = load_manifest()
    points = seed_points(manifest)
    if include_filler:
        points += filler_points(sorted(manifest["tenants"]))

    client = qdrant()
    if client.collection_exists(config.TENANT_COLLECTION):
        client.delete_collection(config.TENANT_COLLECTION)
    client.create_collection(config.TENANT_COLLECTION, vectors_config=models.VectorParams(
        size=len(points[0].vector), distance=models.Distance.COSINE))
    for field in INDEXED_FIELDS:
        client.create_payload_index(config.TENANT_COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    for i in range(0, len(points), 256):
        client.upsert(config.TENANT_COLLECTION, points=points[i:i + 256])
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-filler", action="store_true", help="index only the seed company documents")
    args = parser.parse_args()
    n = ingest(include_filler=not args.no_filler)
    print(f"Indexed {n} chunks into '{config.TENANT_COLLECTION}'")
    qdrant().close()


if __name__ == "__main__":
    main()
