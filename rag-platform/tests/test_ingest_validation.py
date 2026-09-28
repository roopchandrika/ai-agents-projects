"""Ingestion refuses chunks that aren't fully protected."""
import pytest

from rag.acl import validate_acl
from rag.tenant_ingest import point

GOOD = {"chunk_id": "t/doc#0", "tenant_id": "northwind", "allowed_roles": ["employee"], "allowed_users": [],
        "classification": "internal"}


def test_valid_acl_passes():
    validate_acl(GOOD)


@pytest.mark.parametrize("change", [
    {"tenant_id": None},
    {"tenant_id": ""},
    {"tenant_id": "   "},
    {"allowed_roles": "employee"},
    {"allowed_roles": [""]},
    {"allowed_roles": [], "allowed_users": []},
    {"classification": "top-secret"},
])
def test_malformed_acl_is_rejected(change):
    with pytest.raises(ValueError):
        validate_acl({**GOOD, **change})


def test_chunk_without_tenant_is_never_built():
    payload = {k: v for k, v in GOOD.items() if k != "tenant_id"}
    with pytest.raises(ValueError, match="tenant_id"):
        point("t/doc#0", [0.0] * 384, payload)
