"""Computes the USD cost of one model call from its token usage.

Grounded in a real observed `usage_metadata` shape (not assumed): the
`cache_read` and `cache_creation` counts LangChain reports are a breakdown
*of* `input_tokens`, not additional tokens on top of it.
"""

from __future__ import annotations

from pos import config


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
) -> float:
    """Computes the USD cost of one model call.

    Args:
        input_tokens: Total input tokens for the call, as reported in
            `usage_metadata["input_tokens"]`. Includes `cache_read` and
            `cache_creation` tokens -- they are not additional.
        output_tokens: Output tokens for the call.
        cache_read: Input tokens served from the prompt cache.
        cache_creation: Input tokens newly written to the prompt cache.

    Returns:
        The call's cost in US dollars.
    """
    tier = config.PRICE_LONG if input_tokens > config.LONG_CONTEXT_THRESHOLD else config.PRICE_SHORT
    uncached = input_tokens - cache_read - cache_creation

    return (
        uncached * tier.input
        + cache_read * tier.cached
        + cache_creation * tier.cache_write
        + output_tokens * tier.output
    ) / 1_000_000
