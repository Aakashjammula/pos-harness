"""What a tool is, as far as the rest of the app is concerned.

Every tool lives in its own module and describes itself with one ToolSpec.
Nothing else in the codebase names a specific tool: the credential form
fields, the /tools API, the per-user on/off switches and the tools bound to
the LLM are all driven by the registry (see registry.py). Adding a tool is
adding one file here plus one import in __init__.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class ToolSpec:
    id: str
    """Stable identifier used in the API, the database and query strings."""
    label: str
    description: str
    build: Callable[[Mapping[str, str]], BaseTool]
    """Construct the tool from an env mapping. Only called when available()."""
    credential_provider: str | None = None
    """Key in the credential store, e.g. "tavily". None = needs no credentials."""
    credential_fields: Mapping[str, str] = field(default_factory=dict)
    """Form field name -> environment variable, e.g. {"tavily_api_key": "TAVILY_API_KEY"}."""
    default_enabled: bool = True
    """Whether a user who never touched the switch gets this tool (still
    subject to available())."""

    @property
    def requires_key(self) -> bool:
        return bool(self.credential_fields)

    def available(self, env: Mapping[str, str]) -> bool:
        """True when every credential this tool needs is present in env."""
        return all(env.get(var) for var in self.credential_fields.values())
