"""Access-control model: who a user is, what a chunk's ACL looks like, and how the two are matched.

The same rule is expressed twice on purpose: as a Qdrant filter applied inside the vector search, and as a
plain Python check run on every chunk afterwards. A bug in one should not become a data leak.
"""
from dataclasses import dataclass

from qdrant_client import models

ACL_FIELDS = ("tenant_id", "allowed_roles", "allowed_users", "classification")
CLASSIFICATIONS = ("public", "internal", "confidential", "restricted")


@dataclass(frozen=True)
class User:
    id: str
    tenant_id: str
    roles: tuple[str, ...]


def validate_acl(payload: dict) -> None:
    """Raise if a chunk's ACL is missing or malformed. Ingestion calls this for every chunk,
    so an unprotected chunk can never be written."""
    tenant = payload.get("tenant_id")
    if not isinstance(tenant, str) or not tenant.strip():
        raise ValueError(f"chunk {payload.get('chunk_id')!r} has no tenant_id")
    for field in ("allowed_roles", "allowed_users"):
        value = payload.get(field)
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise ValueError(f"chunk {payload.get('chunk_id')!r}: {field} must be a list of non-empty strings")
    if not payload["allowed_roles"] and not payload["allowed_users"]:
        raise ValueError(f"chunk {payload.get('chunk_id')!r} is readable by nobody")
    if payload.get("classification") not in CLASSIFICATIONS:
        raise ValueError(f"chunk {payload.get('chunk_id')!r}: unknown classification")


def acl_filter(user: User) -> models.Filter:
    """Pre-filter for the vector search: same tenant AND (a shared role OR the user is named)."""
    grants = [models.FieldCondition(key="allowed_users", match=models.MatchValue(value=user.id))]
    if user.roles:
        grants.append(models.FieldCondition(key="allowed_roles", match=models.MatchAny(any=list(user.roles))))
    return models.Filter(
        must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=user.tenant_id))],
        should=grants,  # at least one must match
    )


def is_authorized(user: User, payload: dict) -> bool:
    """The application-side check, independent of the vector store."""
    if payload.get("tenant_id") != user.tenant_id:
        return False
    return bool(set(user.roles) & set(payload.get("allowed_roles") or [])) or \
        user.id in (payload.get("allowed_users") or [])
