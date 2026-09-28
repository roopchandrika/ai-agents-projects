"""JWT identity. The tenant and roles used for filtering come only from a verified token,
never from the request body."""
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config
from .acl import User

_bearer = HTTPBearer(auto_error=False)


def _secret() -> str:
    if len(config.JWT_SECRET) < 32:
        raise RuntimeError("Set JWT_SECRET in .env to a random string of at least 32 characters")
    return config.JWT_SECRET


def issue_token(user: User, ttl_minutes: int = config.TOKEN_TTL_MINUTES) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user.id, "tenant_id": user.tenant_id, "roles": list(user.roles),
              "iss": config.JWT_ISSUER, "iat": now, "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(claims, _secret(), algorithm="HS256")


def verify_token(token: str) -> User:
    claims = jwt.decode(token, _secret(), algorithms=["HS256"], issuer=config.JWT_ISSUER,
                        options={"require": ["sub", "tenant_id", "roles", "exp", "iss"]})
    if not isinstance(claims["roles"], list) or not claims["tenant_id"]:
        raise jwt.InvalidTokenError("malformed claims")
    return User(id=claims["sub"], tenant_id=claims["tenant_id"], roles=tuple(claims["roles"]))


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return verify_token(creds.credentials)
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})


def require_admin(user: User = Depends(current_user)) -> User:
    if "admin" not in user.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user
