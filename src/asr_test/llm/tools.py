"""Tools bound to LangChainLlm via bind_tools(). Kept separate from
langchain_llm.py so the tool list can grow without that file's stream()
logic getting harder to read."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool


@tool
def get_current_time() -> str:
    """Return the current date and time (UTC). Use this whenever the
    user asks what day/time it is or something that depends on it."""
    return datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M UTC")


def default_tools() -> list[BaseTool]:
    """get_current_time always; web search only if TAVILY_API_KEY is set
    — langchain_tavily raises at construction time otherwise, and a
    voice agent shouldn't fail to start just because search isn't
    configured."""
    tools: list[BaseTool] = [get_current_time]
    if os.environ.get("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
    else:
        print("  TAVILY_API_KEY not set — web search tool disabled")
    return tools
