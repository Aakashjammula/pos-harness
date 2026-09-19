"""Web search via Tavily. Needs a Tavily API key (https://tavily.com)."""

from __future__ import annotations

from collections.abc import Mapping

from langchain_core.tools import BaseTool

from .base import ToolSpec
from .registry import register


def _build(env: Mapping[str, str]) -> BaseTool:
    # Imported here so listing tools (GET /tools, /options) never pulls in
    # langchain_tavily. The key is passed explicitly rather than through
    # os.environ, which would leak one user's key into every other session.
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=3, tavily_api_key=env["TAVILY_API_KEY"])


SPEC = register(
    ToolSpec(
        id="web_search",
        label="Web search",
        description="Lets the assistant search the web for current facts. Needs your own Tavily API key.",
        build=_build,
        credential_provider="tavily",
        credential_fields={"tavily_api_key": "TAVILY_API_KEY"},
    )
)
