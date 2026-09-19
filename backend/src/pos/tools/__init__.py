"""Tools the assistant can call. See base.py for the contract.

Importing this package imports each tool module purely for its
register(...) side effect, the same way pos.llm.providers works."""

from . import current_time, web_search  # noqa: F401  (registration side effects)
from .base import ToolSpec
from .registry import all_tools, build_tools, credential_fields, get_tool, register, resolve_enabled, tool_status

__all__ = [
    "ToolSpec",
    "all_tools",
    "build_tools",
    "credential_fields",
    "get_tool",
    "register",
    "resolve_enabled",
    "tool_status",
]
