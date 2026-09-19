"""Current date and time. Needs no credentials."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from langchain_core.tools import BaseTool, tool

from .base import ToolSpec
from .registry import register


@tool
def get_current_time() -> str:
    """Return the current date and time (UTC). Use this whenever the
    user asks what day/time it is or something that depends on it."""
    return datetime.now(UTC).strftime("%A, %Y-%m-%d %H:%M UTC")


def _build(env: Mapping[str, str]) -> BaseTool:
    return get_current_time


SPEC = register(
    ToolSpec(
        id="get_current_time",
        label="Current time",
        description="Lets the assistant look up today's date and time instead of guessing.",
        build=_build,
    )
)
