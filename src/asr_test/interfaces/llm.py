from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one turn as text pieces.

    Fully stateless — callers pass the full prior-turns message list
    each call, and implementations own no conversation history *and no
    per-call metrics as instance state*. Timing (ttft, total) is the
    caller's job to measure around the stream() call, not something
    read back from the engine afterward — one instance can be shared
    across concurrent sessions (see server.py's provider cache), and a
    shared mutable last_ttft/last_total would race between them.

    `usage`, if passed, is a caller-owned dict (same pattern as
    `cancel`) that an implementation may fill in place with per-call
    usage/cost data once the stream finishes — not stored on self, so
    sharing one instance across concurrent sessions stays race-free.
    An implementation that doesn't track usage (or a call where the
    backend never reported it) simply leaves it untouched — callers
    must treat a missing key as "unknown," never assume zero.
    """

    @abstractmethod
    def stream(
        self, messages: list[dict], cancel: threading.Event, usage: dict | None = None
    ) -> Iterator[str]: ...
