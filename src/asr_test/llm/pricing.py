"""Best-effort LLM pricing lookup. LLM prices change over time and this
table is a point-in-time snapshot, not a guarantee — override any entry
with {PROVIDER}_PRICE_INPUT_PER_1K / {PROVIDER}_PRICE_OUTPUT_PER_1K env
vars (provider name upper-cased) if it's gone stale. Local is always
free and never consults either the table or an override."""

from __future__ import annotations

import os

PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model) -> (input $/1K tokens, output $/1K tokens)
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
}


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    if provider == "local":
        return (0.0, 0.0)

    env_in = os.environ.get(f"{provider.upper()}_PRICE_INPUT_PER_1K")
    env_out = os.environ.get(f"{provider.upper()}_PRICE_OUTPUT_PER_1K")
    if env_in is not None and env_out is not None:
        return (float(env_in), float(env_out))

    return PRICING.get((provider, model))


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    prices = price_for(provider, model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
