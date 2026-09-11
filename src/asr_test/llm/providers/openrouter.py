"""OpenRouter (routes to many providers/models behind one API). See
https://docs.langchain.com/oss/python/langchain/models#openrouter."""

from __future__ import annotations

import os

from langchain_openrouter import ChatOpenRouter

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "openrouter/auto"


@register
class OpenRouterProvider(LlmProviderBase):
    name = "openrouter"
    priority = 50

    def detect(self, env) -> bool:
        return bool(env.get("OPENROUTER_API_KEY"))

    def resolve(self, model_override: str | None, env) -> ProviderConfig:
        api_key = env["OPENROUTER_API_KEY"]
        model = model_override or env.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
        return ProviderConfig(name=self.name, model=model, api_key=api_key)

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return ChatOpenRouter(model=provider.model, api_key=provider.api_key, **model_kwargs)

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("OPENROUTER_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("OPENROUTER_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return None

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("OPENROUTER_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return None
