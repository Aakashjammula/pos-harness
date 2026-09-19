"""One LLM turn: stream text pieces, time them, collect usage.

Shared by voice (Agent.respond, which cuts the pieces into sentences for TTS)
and text (POST /chat/stream, which forwards each piece as a server-sent event),
so both run the identical LlmBase.stream() path -- content handling, tool
calls, cancellation and usage reporting live in one place."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pos.interfaces import LlmBase


@dataclass
class TurnStats:
    text: str            # everything that arrived, even if the turn was stopped early
    usage: dict          # filled by the LLM; empty when the backend reported none
    ttft: float | None   # seconds to the first piece; None if nothing arrived
    total: float | None  # seconds from start to the last piece; None if nothing arrived
    stopped: bool        # on_piece asked to stop before the stream finished


def run_turn(
    llm: LlmBase,
    messages: list[dict],
    cancel: threading.Event,
    on_piece: Callable[[str], bool | None],
) -> TurnStats:
    """Stream one reply. `on_piece` sees each text piece as it arrives; returning
    False stops the turn (the generator is abandoned, `stopped=True`). Errors
    from the LLM propagate to the caller, which decides how to report them."""
    usage: dict = {}
    parts: list[str] = []
    ttft: float | None = None
    stopped = False
    started = time.perf_counter()
    for piece in llm.stream(messages, cancel, usage):
        if ttft is None:
            ttft = time.perf_counter() - started
        parts.append(piece)
        if on_piece(piece) is False:
            stopped = True
            break
    total = time.perf_counter() - started if ttft is not None else None
    return TurnStats("".join(parts), usage, ttft, total, stopped)
