"""The `pos` command: runs the server and opens the app.

Packaged, there is one process on one port -- FastAPI serves both the API
and the built UI. Running the two dev servers separately (`npm run dev`
plus uvicorn) still works and is unaffected by anything here.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main() -> None:
    """Starts the server and opens a browser at it."""
    parser = argparse.ArgumentParser(prog="pos", description="Run the pos agent locally.")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default: 8000)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default: 127.0.0.1 -- this machine only)",
    )
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # A timer, not a call: uvicorn.run blocks, so the browser has to be
        # opened from a thread, and a moment's delay lets the server bind
        # before the first request arrives.
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    print(f"pos is running at {url}")
    uvicorn.run("pos.app:app", host=args.host, port=args.port, log_level="info")
