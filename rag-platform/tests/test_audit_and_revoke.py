"""Audit log contents, append-only enforcement, and permission revocation taking effect immediately."""
import sqlite3

import pytest

from rag import audit
from rag.secure import secure_retrieve

REVIEW = "northwind/perf-review-alice-chen"


def test_retrievals_are_logged_and_queryable(client, auth, users):
    secure_retrieve("Northwind salary bands 2026", users["nw-bob"])
    r = client.get("/admin/documents/northwind/salary-bands-2026/access-log", headers=auth("nw-dave"))
    assert r.status_code == 200
    assert "nw-bob" in [u["user_id"] for u in r.json()["users"]]


def test_access_log_is_scoped_to_the_admins_tenant(client, auth, users):
    secure_retrieve("Helix salary bands 2026", users["hx-frank"])
    r = client.get("/admin/documents/helix/salary-bands-2026/access-log", headers=auth("nw-dave"))
    assert r.json()["users"] == []


@pytest.mark.parametrize("sql", [
    "UPDATE retrievals SET user_id = 'someone-else'",
    "DELETE FROM retrievals",
    "DELETE FROM retrieved_chunks",
    "UPDATE acl_changes SET new_acl = '{}'",
    "DELETE FROM security_alerts",
])
def test_audit_log_is_append_only(sql, users):
    # Triggers fire per row, so make sure every table has one.
    secure_retrieve("handbook", users["nw-alice"])
    audit.log_acl_change(users["nw-dave"], "northwind/test-doc", {}, {})
    audit.log_alert(None, users["nw-alice"], "test alert")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with audit.connect() as conn:
            conn.execute(sql)


def test_revoke_takes_effect_on_the_next_query(client, auth, users):
    alice = users["nw-alice"]
    title = "Performance Review: Alice Chen"
    assert REVIEW in {h.doc_id for h in secure_retrieve(title, alice)}

    revoke = {"allowed_roles": ["hr"], "allowed_users": [], "classification": "restricted"}
    restore = {"allowed_roles": ["hr"], "allowed_users": ["nw-alice"], "classification": "restricted"}
    try:
        r = client.put(f"/admin/documents/{REVIEW}/acl", headers=auth("nw-dave"), json=revoke)
        assert r.status_code == 200
        assert REVIEW not in {h.doc_id for h in secure_retrieve(title, alice, k=20)}
        assert REVIEW in {h.doc_id for h in secure_retrieve(title, users["nw-bob"])}  # HR still has it
    finally:
        client.put(f"/admin/documents/{REVIEW}/acl", headers=auth("nw-dave"), json=restore)

    with audit.connect() as conn:
        changes = conn.execute("SELECT * FROM acl_changes WHERE doc_id = ?", (REVIEW,)).fetchall()
    assert len(changes) >= 2
