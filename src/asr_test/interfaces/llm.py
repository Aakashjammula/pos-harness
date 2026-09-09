from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one turn as text pieces.

    Fully stateless — callers pass the full prior-turns message list
    each call, and implementations own no conversation history *and no
    per-call metrics*. Timing (ttft, total) is the caller's job to
    measure around the stream() call, not something read back from the
    engine afterward — one instance can be shared across concurrent
    sessions (see server.py's provider cache), and a shared mutable
    last_ttft/last_total would race between them.
    """

    @abstractmethod
    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]: ...
