"""Short-lived JWT access tokens plus opaque refresh tokens.

Only the sha256 of a refresh token is ever stored, so a database leak
yields nothing usable -- same reasoning as password_hash."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from pos import config

ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=30)
MAGIC_LINK_TOKEN_TTL = timedelta(minutes=15)
_ALGORITHM = "HS256"


def create_access_token(user_id: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + ACCESS_TOKEN_TTL},
        config.JWT_SECRET,
        algorithm=_ALGORITHM,
    )


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_refresh_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_magic_link_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_magic_link_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(32)
    return raw, hash_magic_link_token(raw)
