"""Dispatch layer over whatever providers are registered -- see base.py
for the contract each provider implements. Providers self-register via
@register when their module is imported (see providers/__init__.py,
which imports local.py/openai.py/azure.py for exactly this side effect)."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .base import LlmProviderBase, ProviderConfig

_REGISTRY: dict[str, LlmProviderBase] = {}


def register(cls: type[LlmProviderBase]) -> type[LlmProviderBase]:
    _REGISTRY[cls.name] = cls()
    return cls


def resolve_provider(
    model_override: str | None = None, env: Mapping[str, str] | None = None
) -> ProviderConfig:
    """env defaults to the real process environment -- pass an overlay
    (e.g. {**os.environ, "OPENAI_API_KEY": "..."}) to resolve against a
    per-connection override instead, without any provider needing to
    know the difference (see base.py's detect()/resolve())."""
    if env is None:
        env = os.environ
    for provider in sorted(_REGISTRY.values(), key=lambda p: p.priority):
        if provider.detect(env):
            return provider.resolve(model_override, env)
    raise RuntimeError(
        "no LLM provider configured -- set LOCAL_BASE_URL (an OpenAI-compatible server) "
        "or a provider API key such as OPENAI_API_KEY"
    )


def is_configured(env: Mapping[str, str] | None = None) -> bool:
    """True if some provider detects its settings in env. Cheap: never
    resolves a provider, so it makes no network request."""
    if env is None:
        env = os.environ
    return any(p.detect(env) for p in _REGISTRY.values())


def build_model(provider: ProviderConfig, **model_kwargs):
    return _REGISTRY[provider.name].build_model(provider, **model_kwargs)


def price_for(provider: ProviderConfig) -> tuple[float, float] | None:
    return _REGISTRY[provider.name].price_for(provider)


def context_window_for(provider: ProviderConfig) -> int | None:
    return _REGISTRY[provider.name].context_window_for(provider)


def estimate_cost(provider: ProviderConfig, input_tokens: int, output_tokens: int) -> float | None:
    prices = price_for(provider)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


def call_kwargs(provider: ProviderConfig) -> dict:
    return _REGISTRY[provider.name].call_kwargs(provider)
