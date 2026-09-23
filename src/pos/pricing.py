"""Computes the USD cost of one model call from its token usage.

Rates come from the model catalogue (see models.py), so switching model or
provider switches the arithmetic with it. `.env` still overrides any single
value, for a deployment the catalogue has never heard of.

Grounded in a real observed `usage_metadata` shape (not assumed): the
`cache_read` and `cache_creation` counts LangChain reports are a breakdown
*of* `input_tokens`, not additional tokens on top of it.
"""

from __future__ import annotations

import dataclasses
import functools

from pos import config, models

# Used only when neither the catalogue nor `.env` says anything -- an
# unrecognised deployment on an unreachable network. Zero rather than a
# guess: a cost of $0.00 is visibly wrong, while a plausible-looking wrong
# number is not.
_UNKNOWN = models.Rates(input=0.0, cached=0.0, cache_write=0.0, output=0.0)


@dataclasses.dataclass(frozen=True)
class Plan:
    """The numbers needed to price and size one model.

    Attributes:
        short: Rates below `long_threshold` input tokens.
        long: Rates at or above it.
        long_threshold: Where the second tier starts; None for one tier.
        context_window: Total tokens the model holds, or None if unknown.
        known: False when nothing was found and the rates are all zero.
    """

    short: models.Rates
    long: models.Rates
    long_threshold: int | None
    context_window: int | None
    known: bool


def _apply(rates: models.Rates, override: dict[str, float | None]) -> models.Rates:
    """Returns `rates` with any set override values replacing its own."""
    return models.Rates(
        input=override["input"] if override["input"] is not None else rates.input,
        cached=override["cached"] if override["cached"] is not None else rates.cached,
        cache_write=override["cache_write"] if override["cache_write"] is not None else rates.cache_write,
        output=override["output"] if override["output"] is not None else rates.output,
    )


@functools.lru_cache(maxsize=32)
def plan_for(spec: str, reported: str | None = None) -> Plan:
    """Everything needed to price and size one model.

    Cached: the catalogue itself is already cached on disk, and this saves
    re-deriving the same answer on every turn.

    Args:
        spec: The model as configured -- "openai:gpt-5", or a bare name for
            whichever provider the keys point at. Its provider decides which
            price table is read, since the two differ.
        reported: The model id the reply carried, tried when `spec`'s name
            is an Azure deployment the catalogue has never heard of.

    Returns:
        A Plan. `known` is False when the model is unrecognised and nothing
        in `.env` filled the gap.
    """
    provider, name = config.split_model(spec)
    info = models.lookup(provider, name)
    if info.short is None and reported:
        info = models.lookup(provider, reported)
    short = info.short or _UNKNOWN
    long_rates = info.long or short

    short = _apply(short, config.PRICE_SHORT_OVERRIDE)
    long_rates = _apply(long_rates, config.PRICE_LONG_OVERRIDE)

    threshold = config.LONG_CONTEXT_THRESHOLD_OVERRIDE or info.long_threshold
    window = config.CONTEXT_WINDOW_OVERRIDE or info.context_window
    known = info.short is not None or any(v is not None for v in config.PRICE_SHORT_OVERRIDE.values())

    return Plan(
        short=short,
        long=long_rates,
        long_threshold=threshold,
        context_window=window,
        known=known,
    )


def cost_usd(
    spec: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
) -> float:
    """Computes the USD cost of one model call.

    Args:
        spec: The model as configured, used to look up its rates.
        input_tokens: Total input tokens for the call, as reported in
            `usage_metadata["input_tokens"]`. Includes `cache_read` and
            `cache_creation` tokens -- they are not additional.
        output_tokens: Output tokens for the call.
        cache_read: Input tokens served from the prompt cache.
        cache_creation: Input tokens newly written to the prompt cache.

    Returns:
        The call's cost in US dollars, or 0.0 for a model whose rates are
        unknown.
    """
    plan = plan_for(spec)
    over_threshold = plan.long_threshold is not None and input_tokens > plan.long_threshold
    rates = plan.long if over_threshold else plan.short
    uncached = input_tokens - cache_read - cache_creation

    return (
        uncached * rates.input
        + cache_read * rates.cached
        + cache_creation * rates.cache_write
        + output_tokens * rates.output
    ) / 1_000_000
