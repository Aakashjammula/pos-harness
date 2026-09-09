from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one user turn as text pieces.

    Implementations own their own conversation history. `last_ttft` /
    `last_total` reflect the most recently completed `stream()` call and
    are read by the caller for timing telemetry.
    """

    last_ttft: float | None
    last_total: float | None

    @abstractmethod
    def stream(self, user_text: str, cancel: threading.Event) -> Iterator[str]: ...
