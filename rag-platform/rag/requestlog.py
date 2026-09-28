"""Per-request metrics log (SQLite): what every request cost, how long it took and how it was served.
The dashboard reads this table, and eval runs tag their rows with a run id and the judge's verdict."""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT, scope TEXT NOT NULL, user_id TEXT,
    query TEXT NOT NULL, cache_hit INTEGER NOT NULL, cache_similarity REAL, route TEXT, model TEXT,
    escalated INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL, retrieval_ms REAL, generation_ms REAL, total_ms REAL NOT NULL,
    top_score REAL, cited INTEGER, correct INTEGER);
CREATE INDEX IF NOT EXISTS requests_run ON requests (run_id);
"""


@dataclass
class RequestRecord:
    request_id: str
    scope: str                       # "public" or a tenant id
    query: str
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    total_ms: float
    user_id: str | None = None
    run_id: str | None = None
    cache_similarity: float | None = None
    route: str | None = None         # "simple" / "complex" / None when routing is off
    model: str | None = None         # model that produced the final answer (None for cache hits)
    escalated: bool = False
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    top_score: float | None = None
    cited: bool | None = None


@contextmanager
def connect():
    config.REQUEST_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.REQUEST_DB)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


def log(rec: RequestRecord) -> None:
    row = dict(rec.__dict__, ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    cols = ", ".join(row)
    with connect() as conn:
        conn.execute(f"INSERT INTO requests ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def set_correct(request_id: str, correct: bool) -> None:
    with connect() as conn:
        conn.execute("UPDATE requests SET correct = ? WHERE request_id = ?", (int(correct), request_id))
