"""The set of tools that exist, and the operations the app needs on them."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from langchain_core.tools import BaseTool

from .base import ToolSpec

_TOOLS: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.id in _TOOLS:
        raise ValueError(f"duplicate tool id {spec.id!r}")
    _TOOLS[spec.id] = spec
    return spec


def all_tools() -> list[ToolSpec]:
    return list(_TOOLS.values())


def get_tool(tool_id: str) -> ToolSpec | None:
    return _TOOLS.get(tool_id)


def credential_fields() -> dict[str, dict[str, str]]:
    """{credential provider -> {form field -> env var}} for every tool that
    needs a key. The credential routes are built from this, so a new tool's
    key form works without touching them."""
    return {
        spec.credential_provider: dict(spec.credential_fields)
        for spec in _TOOLS.values()
        if spec.credential_provider and spec.credential_fields
    }


def resolve_enabled(settings: Mapping[str, bool], env: Mapping[str, str] | None = None) -> list[str]:
    """Ids of the tools that will actually be bound: the user's own switch
    where they set one, the tool's default otherwise, and only if its
    credentials are present in env."""
    if env is None:
        env = os.environ
    return [
        spec.id
        for spec in _TOOLS.values()
        if settings.get(spec.id, spec.default_enabled) and spec.available(env)
    ]


def build_tools(enabled: Iterable[str] | None = None, env: Mapping[str, str] | None = None) -> list[BaseTool]:
    """Construct the tools for `enabled` ids (None = every default-enabled
    tool). Unknown ids and tools whose credentials are missing are skipped,
    never raised, so a session still starts without them."""
    if env is None:
        env = os.environ
    ids = resolve_enabled({}, env) if enabled is None else list(enabled)
    tools: list[BaseTool] = []
    for tool_id in ids:
        spec = _TOOLS.get(tool_id)
        if spec is not None and spec.available(env):
            tools.append(spec.build(env))
    return tools


def tool_status(env: Mapping[str, str] | None = None, settings: Mapping[str, bool] | None = None) -> list[dict]:
    """Declarative status for /options: which tools would be bound. Never
    constructs a tool, so it stays cheap to call on every page load."""
    if env is None:
        env = os.environ
    active = set(resolve_enabled(settings or {}, env))
    return [{"name": s.id, "label": s.label, "enabled": s.id in active} for s in _TOOLS.values()]
