"""Embedding model and Qdrant client, shared by ingestion, the API and the eval scripts."""
import threading
from functools import lru_cache

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from . import config

# The embedded Qdrant store isn't built for concurrent access; hold this around every search (a few ms).
store_lock = threading.Lock()


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    # Embedded mode: no server needed, but only one process can open the folder at a time.
    return QdrantClient(path=str(config.QDRANT_PATH))


def passage_text(title: str, text: str) -> str:
    """What gets embedded for a chunk. The title gives short chunks their context."""
    return f"{title}\n\n{text}"


def embed_passages(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, normalize_embeddings=True, batch_size=32,
                             show_progress_bar=len(texts) > 100).tolist()


def embed_query(query: str) -> list[float]:
    return embedder().encode(config.QUERY_PREFIX + query, normalize_embeddings=True).tolist()


def reset_collection(name: str, vectors_config: models.VectorParams | None = None) -> None:
    """Empty a collection and (optionally) recreate it. Points are deleted before the collection is dropped
    because embedded Qdrant brings a dropped collection's points back if it is recreated under the same name
    in the same process."""
    client = qdrant()
    if client.collection_exists(name):
        client.delete(name, points_selector=models.FilterSelector(filter=models.Filter()), wait=True)
        client.delete_collection(name)
    if vectors_config is not None:
        client.create_collection(name, vectors_config=vectors_config)
