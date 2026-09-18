"""Cookie handling and the current-user dependency.

The access token travels in an httpOnly cookie rather than a header or
localStorage: script-readable storage hands the token to any injected
script, and a query string would land in uvicorn's access log."""

from __future__ import annotations

from fastapi import HTTPException, Request, Response

from pos import config
from pos.auth.tokens import ACCESS_TOKEN_TTL, REFRESH_TOKEN_TTL, decode_access_token

ACCESS_COOKIE = "pos_access"
REFRESH_COOKIE = "pos_refresh"


def _set(response: Response, name: str, value: str, max_age: int) -> None:
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    _set(response, ACCESS_COOKIE, access_token, int(ACCESS_TOKEN_TTL.total_seconds()))
    _set(response, REFRESH_COOKIE, refresh_token, int(REFRESH_TOKEN_TTL.total_seconds()))


def clear_auth_cookies(response: Response) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE):
        response.delete_cookie(name, path="/")


def user_id_from_request(request: Request) -> str | None:
    token = request.cookies.get(ACCESS_COOKIE)
    return decode_access_token(token) if token else None


def require_user_id(request: Request) -> str:
    user_id = user_id_from_request(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user_id
