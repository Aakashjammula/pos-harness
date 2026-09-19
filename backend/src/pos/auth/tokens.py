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


def create_access_token(user_id: str, session_id: str | None = None) -> str:
    """`session_id` binds the token to a server-side login session, which is
    what makes logout and "sign out that device" take effect immediately."""
    now = datetime.now(UTC)
    claims = {"sub": user_id, "iat": now, "exp": now + ACCESS_TOKEN_TTL}
    if session_id:
        claims["sid"] = session_id
    return jwt.encode(claims, config.JWT_SECRET, algorithm=_ALGORITHM)


def decode_access_claims(token: str) -> tuple[str, str | None] | None:
    """(user id, session id or None) for a valid token, else None."""
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return None
    subject, sid = payload.get("sub"), payload.get("sid")
    if not isinstance(subject, str):
        return None
    return subject, sid if isinstance(sid, str) else None


def decode_access_token(token: str) -> str | None:
    claims = decode_access_claims(token)
    return claims[0] if claims else None


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
