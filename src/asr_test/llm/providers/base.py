"""The provider-registry contract. See registry.py for the dispatch
functions, and local.py/openai.py/azure.py for the concrete providers
-- see docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md
for why this replaced a scattered if/elif branching approach."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    name: str                        # "local" | "openai" | "azure" | ...
    model: str
    base_url: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None


class LlmProviderBase(ABC):
    """One backend LangChainLlm can talk to. A new provider is one new
    file implementing this and decorated with @register -- no other
    file needs to change (see registry.py's resolve_provider(), which
    iterates whatever's registered rather than branching by name)."""

    name: str
    priority: int   # lower = checked first; the fallback provider uses the highest number

    @abstractmethod
    def detect(self, env: Mapping[str, str]) -> bool:
        """True if this provider's required var(s) are present in env
        -- real os.environ by default, but see registry.resolve_provider()'s
        env= parameter: a per-connection override (e.g. an API key typed
        into the browser's Settings page) can be overlaid on top of it
        without this provider ever knowing the difference."""

    @abstractmethod
    def resolve(self, model_override: str | None, env: Mapping[str, str]) -> ProviderConfig:
        """Build this provider's config. May raise RuntimeError if
        detect() returned True but other required config is missing."""

    @abstractmethod
    def build_model(self, provider: ProviderConfig, **model_kwargs):
        """Construct and return this provider's langchain chat-model instance."""

    @abstractmethod
    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        """(input $/1K tokens, output $/1K tokens), or None if unpriced."""

    @abstractmethod
    def context_window_for(self, provider: ProviderConfig) -> int | None:
        """Max context window in tokens, or None if unknown."""
