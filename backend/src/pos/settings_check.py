"""Refuse to start with a configuration that would be insecure, instead of
finding out at the first request. Messages name the variable, never its value."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from urllib.parse import urlparse

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_MIN_JWT_SECRET = 32


def check_settings(env: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """(fatal errors, warnings) for this environment."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(env.get("JWT_SECRET", "")) < _MIN_JWT_SECRET:
        errors.append(f"JWT_SECRET must be set to at least {_MIN_JWT_SECRET} characters (openssl rand -base64 32)")

    try:
        key = base64.b64decode(env.get("ENCRYPTION_KEY", ""), validate=True)
    except (binascii.Error, ValueError):
        key = b""
    if len(key) != 32:
        errors.append("ENCRYPTION_KEY must be base64 for exactly 32 bytes (openssl rand -base64 32)")

    for previous in (k.strip() for k in env.get("ENCRYPTION_KEY_PREVIOUS", "").split(",") if k.strip()):
        try:
            ok = len(base64.b64decode(previous, validate=True)) == 32
        except (binascii.Error, ValueError):
            ok = False
        if not ok:
            errors.append("ENCRYPTION_KEY_PREVIOUS has an entry that is not base64 for exactly 32 bytes")

    secure = env.get("COOKIE_SECURE", "false").lower() == "true"
    origins = [o.strip() for o in env.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    for origin in origins:
        parsed = urlparse(origin)
        if parsed.scheme == "https" and not secure:
            errors.append(
                f"CORS_ORIGINS includes {origin} (https) but COOKIE_SECURE is not true: "
                "session cookies would also be sent over plain http"
            )
        elif parsed.scheme == "http" and (parsed.hostname or "") not in _LOCAL_HOSTS and not secure:
            warnings.append(
                f"CORS_ORIGINS includes {origin} over plain http and COOKIE_SECURE is false: "
                "session cookies travel unencrypted on that network"
            )
    return errors, warnings
