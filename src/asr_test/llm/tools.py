"""Tools bound to LangChainLlm via bind_tools(). Kept separate from
langchain_llm.py so the tool list can grow without that file's stream()
logic getting harder to read."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime

from langchain_core.tools import BaseTool, tool


@tool
def get_current_time() -> str:
    """Return the current date and time (UTC). Use this whenever the
    user asks what day/time it is or something that depends on it."""
    return datetime.now(UTC).strftime("%A, %Y-%m-%d %H:%M UTC")


def default_tools(env: Mapping[str, str] | None = None) -> list[BaseTool]:
    """get_current_time always; web search only if TAVILY_API_KEY is set
    — langchain_tavily raises at construction time otherwise, and a
    voice agent shouldn't fail to start just because search isn't
    configured. env defaults to the real process environment; pass an
    overlay (e.g. a per-connection key typed into the browser's
    Settings page) to resolve against that instead -- passed explicitly
    to TavilySearch(tavily_api_key=...) rather than mutating os.environ,
    which would leak across every other concurrent session."""
    if env is None:
        env = os.environ
    tools: list[BaseTool] = [get_current_time]
    tavily_key = env.get("TAVILY_API_KEY")
    if tavily_key:
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3, tavily_api_key=tavily_key))
    else:
        print("  TAVILY_API_KEY not set — web search tool disabled")
    return tools


def tool_status(env: Mapping[str, str] | None = None) -> list[dict]:
    """Declarative status for the browser's Connections diagram
    (GET /options) -- mirrors default_tools()'s enablement logic
    without constructing real tool instances (no network calls, no
    langchain_tavily import), so /options stays cheap to call on every
    page load. Same env= override as default_tools()."""
    if env is None:
        env = os.environ
    return [
        {"name": "get_current_time", "label": "Current time", "enabled": True},
        {"name": "web_search", "label": "Web search", "enabled": bool(env.get("TAVILY_API_KEY"))},
    ]
