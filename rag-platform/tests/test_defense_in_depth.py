"""If the vector-store filter breaks, the application-side check still stops the leak and raises an alert."""
from eval.leakage_cases import is_allowed_doc
from rag import acl, audit, secure


def test_second_check_catches_a_broken_filter(monkeypatch, users, manifest):
    user = users["nw-alice"]
    before = len(audit.alerts(user.tenant_id))
    monkeypatch.setattr(acl, "acl_filter", lambda u: None)  # simulate a filter bug: no filtering at all

    hits = secure.secure_retrieve("salary bands and board minutes at every company", user, k=15)

    assert all(is_allowed_doc(user, h.doc_id, manifest) for h in hits)
    assert len(audit.alerts(user.tenant_id)) > before
