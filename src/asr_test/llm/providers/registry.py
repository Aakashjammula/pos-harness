"""Dispatch layer over whatever providers are registered -- see base.py
for the contract each provider implements. Providers self-register via
@register when their module is imported (see providers/__init__.py,
which imports local.py/openai.py/azure.py for exactly this side effect)."""

from __future__ import annotations

from .base import LlmProviderBase, ProviderConfig

_REGISTRY: dict[str, LlmProviderBase] = {}


def register(cls: type[LlmProviderBase]) -> type[LlmProviderBase]:
    _REGISTRY[cls.name] = cls()
    return cls


def resolve_provider(model_override: str | None = None) -> ProviderConfig:
    for provider in sorted(_REGISTRY.values(), key=lambda p: p.priority):
        if provider.detect():
            return provider.resolve(model_override)
    raise RuntimeError(
        "no LLM provider available -- this should be unreachable if a "
        "fallback provider (priority high enough, detect() always True) is registered"
    )


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
