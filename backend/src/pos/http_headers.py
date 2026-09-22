"""Security headers for the packaged app.

When the UI is served by Next.js (development), `next.config.ts` sends
these. A static export can't -- `headers()` is one of the features
`output: 'export'` doesn't support, since there's no Node server left to
send them. So the same headers are sent from here instead, which is also
the only place that can send them once FastAPI is serving the UI itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

# Same policy as next.config.ts. 'unsafe-inline' is required by Next's
# hydration scripts and Tailwind's injected styles; 'unsafe-eval' is not
# included because the packaged build is production, never `next dev`.
# connect-src is 'self' alone: packaged, the API and the UI are one origin.
_CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "media-src 'self' blob:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))

_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(), camera=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def add_security_headers(app: FastAPI) -> None:
    """Attaches the security headers to every response.

    Args:
        app: The app to attach the middleware to.
    """

    @app.middleware("http")
    async def _headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        # setdefault, not assignment: a route that deliberately sets its own
        # policy should keep it.
        for name, value in _HEADERS.items():
            response.headers.setdefault(name, value)
        return response
