"""HTTP-level hardening that is not specific to any one route.

Written as plain ASGI middleware, not Starlette's BaseHTTPMiddleware: that class
buffers and interferes with streaming responses, and /chat/stream is one."""

from __future__ import annotations

import json
from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _normalise(origin: str) -> str:
    return origin.strip().lower().rstrip("/")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


class OriginCheckMiddleware:
    """Refuse browser requests that come from a page we do not serve.

    A browser attaches an Origin header to every cross-origin write and every
    WebSocket handshake, and a web page cannot forge it. So a write or a socket
    whose Origin is present but not one of ours is another site trying to act as
    the signed-in user (CSRF / cross-site WebSocket hijacking). SameSite=Lax
    cookies already stop most of this in current browsers; this covers the
    same-site cases and older ones. Requests with no Origin -- the CLI client,
    curl, tests -- are not browsers and pass. Reads are left to CORS."""

    def __init__(self, app: ASGIApp, allowed_origins: Iterable[str]):
        self.app = app
        self.allowed = {_normalise(o) for o in allowed_origins}

    def _refused(self, scope: Scope) -> bool:
        origin = _header(scope, b"origin")
        return origin is not None and _normalise(origin) not in self.allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket" and self._refused(scope):
            await receive()                                        # the websocket.connect message
            await send({"type": "websocket.close", "code": 1008})  # refuse before accepting
            return
        if scope["type"] == "http" and scope["method"] not in _SAFE_METHODS and self._refused(scope):
            body = json.dumps({"detail": "cross-origin request refused"}).encode()
            await send({
                "type": "http.response.start", "status": 403,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)



# Account data must never sit in a browser or proxy cache (or reappear via the back button).
_NO_STORE_PREFIXES = ("/auth", "/credentials", "/tools", "/sessions")
# Swagger UI / ReDoc load their scripts from a CDN, so a strict CSP would break them.
_DOCS_PATHS = ("/docs", "/redoc")


class SecurityHeadersMiddleware:
    """Baseline response headers on every reply, errors and framework-generated ones
    included. This is an API that returns JSON and event streams, never pages, so the CSP
    is 'default-src none' and nothing may frame it."""

    def __init__(self, app: ASGIApp, *, hsts: bool = False):
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}

                def add(name: bytes, value: str) -> None:
                    if name not in present:          # never override what a route chose on purpose
                        headers.append((name, value.encode("latin-1")))

                add(b"x-content-type-options", "nosniff")
                add(b"referrer-policy", "no-referrer")
                add(b"x-frame-options", "DENY")
                if not path.startswith(_DOCS_PATHS):
                    add(b"content-security-policy", "default-src 'none'; frame-ancestors 'none'")
                if path.startswith(_NO_STORE_PREFIXES):
                    add(b"cache-control", "no-store")
                if self.hsts:
                    add(b"strict-transport-security", "max-age=31536000; includeSubDomains")
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
