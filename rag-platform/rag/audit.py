"""Append-only audit log in SQLite: every retrieval, every ACL change and every security alert.

Triggers reject UPDATE and DELETE, so rows can only ever be added. Swap in Postgres for production;
the schema carries over as is.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config
from .acl import User

TABLES = ("retrievals", "retrieved_chunks", "acl_changes", "security_alerts")

SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, user_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    query TEXT NOT NULL, model TEXT);
CREATE TABLE IF NOT EXISTS retrieved_chunks (
    request_id TEXT NOT NULL REFERENCES retrievals(request_id), ts TEXT NOT NULL, tenant_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL, doc_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS retrieved_chunks_doc ON retrieved_chunks (tenant_id, doc_id, ts);
CREATE TABLE IF NOT EXISTS acl_changes (
    ts TEXT NOT NULL, admin_id TEXT NOT NULL, tenant_id TEXT NOT NULL, doc_id TEXT NOT NULL,
    old_acl TEXT NOT NULL, new_acl TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS security_alerts (
    ts TEXT NOT NULL, request_id TEXT, user_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    chunk_id TEXT, doc_id TEXT, reason TEXT NOT NULL);
""" + "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t} BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t} BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;"""
    for t in TABLES
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def connect():
    """One short-lived connection per operation: commits on success, always closes."""
    config.AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.AUDIT_DB)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


def log_retrieval(request_id: str, user: User, query: str, hits: list, model: str | None) -> None:
    ts = _now()
    with connect() as conn:
        conn.execute("INSERT INTO retrievals VALUES (?, ?, ?, ?, ?, ?)",
                     (request_id, ts, user.id, user.tenant_id, query, model))
        conn.executemany("INSERT INTO retrieved_chunks VALUES (?, ?, ?, ?, ?)",
                         [(request_id, ts, user.tenant_id, h.chunk_id, h.doc_id) for h in hits])


def log_acl_change(admin: User, doc_id: str, old: dict, new: dict) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO acl_changes VALUES (?, ?, ?, ?, ?, ?)",
                     (_now(), admin.id, admin.tenant_id, doc_id, json.dumps(old), json.dumps(new)))


def log_alert(request_id: str | None, user: User, reason: str, chunk_id: str | None = None,
              doc_id: str | None = None) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO security_alerts VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (_now(), request_id, user.id, user.tenant_id, chunk_id, doc_id, reason))


def who_retrieved(tenant_id: str, doc_id: str, days: int = 30) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds")
    with connect() as conn:
        rows = conn.execute(
            """SELECT r.user_id, COUNT(DISTINCT r.request_id) AS retrievals, MIN(r.ts) AS first_ts,
                      MAX(r.ts) AS last_ts
               FROM retrieved_chunks c JOIN retrievals r USING (request_id)
               WHERE c.tenant_id = ? AND c.doc_id = ? AND c.ts >= ?
               GROUP BY r.user_id ORDER BY last_ts DESC""",
            (tenant_id, doc_id, since)).fetchall()
    return [dict(r) for r in rows]


def alerts(tenant_id: str, days: int = 30) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds")
    with connect() as conn:
        rows = conn.execute("SELECT * FROM security_alerts WHERE tenant_id = ? AND ts >= ? ORDER BY ts DESC",
                            (tenant_id, since)).fetchall()
    return [dict(r) for r in rows]
