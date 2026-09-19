"""Cookie handling and the current-user dependency.

The access token travels in an httpOnly cookie rather than a header or
localStorage: script-readable storage hands the token to any injected
script, and a query string would land in uvicorn's access log."""

from __future__ import annotations

from fastapi import HTTPException, Request, Response
from starlette.requests import HTTPConnection

from pos import config
from pos.auth.store import UserStore
from pos.auth.tokens import ACCESS_TOKEN_TTL, REFRESH_TOKEN_TTL, decode_access_claims

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


class Auth:
    """Who is making this request. A valid access token is not enough: it must
    also name a login session that has not been revoked, which is what makes
    logout and "sign out that device" real instead of waiting out the token."""

    def __init__(self, users: UserStore):
        self._users = users

    def _claims(self, conn: HTTPConnection) -> tuple[str, str] | None:
        token = conn.cookies.get(ACCESS_COOKIE)
        claims = decode_access_claims(token) if token else None
        if claims is None or claims[1] is None:      # no session id: a token from before sessions existed
            return None
        user_id, session_id = claims
        return (user_id, session_id) if self._users.auth_session_active(session_id, user_id) else None

    def user_id_from_request(self, conn: HTTPConnection) -> str | None:
        claims = self._claims(conn)
        return claims[0] if claims else None

    def session_id_from_request(self, conn: HTTPConnection) -> str | None:
        claims = self._claims(conn)
        return claims[1] if claims else None

    async def require_user_id(self, request: Request) -> str:
        user_id = self.user_id_from_request(request)
        if user_id is None:
            raise HTTPException(status_code=401, detail="not authenticated")
        return user_id
