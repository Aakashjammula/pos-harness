"""OpenAI's own API. See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "gpt-4o-mini"

# Best-effort snapshot, not guaranteed current -- override via
# OPENAI_PRICE_INPUT_PER_1K/OPENAI_PRICE_OUTPUT_PER_1K if these have
# changed (they will, eventually).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# Same caveat as PRICING -- override via OPENAI_CONTEXT_WINDOW.
CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
}


@register
class OpenAIProvider(LlmProviderBase):
    name = "openai"
    priority = 10

    def detect(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def resolve(self, model_override: str | None) -> ProviderConfig:
        api_key = os.environ["OPENAI_API_KEY"]
        model = model_override or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
        return ProviderConfig(name=self.name, model=model, api_key=api_key)

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return ChatOpenAI(
            base_url=None,
            api_key=provider.api_key,
            model=provider.model,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("OPENAI_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("OPENAI_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return PRICING.get(provider.model)

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("OPENAI_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return CONTEXT_WINDOWS.get(provider.model)
