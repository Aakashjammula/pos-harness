from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LlmBase(ABC):
    """Chat LLM: streams a response to one turn as text pieces.
    Stateless — callers pass the full prior-turns message list each
    call; implementations own no conversation history. `last_ttft` /
    `last_total` reflect the most recently completed `stream()` call.
    """

    last_ttft: float | None
    last_total: float | None

    @abstractmethod
    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]: ...
