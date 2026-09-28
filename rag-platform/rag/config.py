"""Central configuration. Everything tunable lives here and can be overridden in .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Paths
CORPUS_DIR = ROOT / "data" / "corpus"
QDRANT_PATH = ROOT / "data" / "qdrant"          # used when QDRANT_URL is empty (embedded mode)
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Models
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")
EVAL_GEN_MODEL = os.getenv("EVAL_GEN_MODEL", "claude-sonnet-5")

# Embeddings. bge-small has a 512-token limit, so chunks stay a little under it.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

# Chunking and retrieval
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Vector store
QDRANT_URL = os.getenv("QDRANT_URL", "")        # e.g. http://localhost:6333 when using docker compose
COLLECTION = os.getenv("COLLECTION", "docs")

# Multi-tenant service (Project 2)
TENANT_DIR = ROOT / "data" / "tenants"
TENANT_COLLECTION = os.getenv("TENANT_COLLECTION", "tenant_chunks")
AUDIT_DB = Path(os.getenv("AUDIT_DB", ROOT / "data" / "audit.db"))
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ISSUER = "rag-platform"
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "60"))

# Semantic cache and routing (Project 1)
SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-4-5")
LARGE_MODEL = os.getenv("LARGE_MODEL", ANSWER_MODEL)
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "semantic_cache")
# Embedding similarity only nominates a candidate (it can't tell "Pioneer 10" from "Pioneer 11"); a small model then
# confirms both questions have the same answer. Both tuned with eval/tune_threshold.py.
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.87"))
CACHE_VERIFY_MODEL = os.getenv("CACHE_VERIFY_MODEL", SMALL_MODEL)
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "168"))
REQUEST_DB = Path(os.getenv("REQUEST_DB", ROOT / "data" / "requests.db"))

# USD per 1M tokens (input, output). Used to log cost per request.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
