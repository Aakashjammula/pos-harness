"""Settings read from the environment.

Single local user, no database of credentials: every setting here comes
from `.env`, edited by hand. See `.env.example` for the full list.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclasses.dataclass(frozen=True)
class PriceTier:
    """Per-1M-token USD prices for one context-length tier.

    Attributes:
        input: Price for input tokens that were neither cached nor newly
            written to the cache.
        cached: Price for input tokens read from the prompt cache.
        cache_write: Price for input tokens newly written to the cache.
        output: Price for output tokens.
    """

    input: float
    cached: float
    cache_write: float
    output: float


def _float_env(name: str, default: float) -> float:
    """Reads a float from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _int_env(name: str, default: int) -> int:
    """Reads an int from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _optional_float_env(name: str) -> float | None:
    """Reads a float from the environment, or None when it isn't set."""
    raw = os.environ.get(name)
    return float(raw) if raw else None


# Which provider to call, as `init_chat_model` names them: "azure_openai" or
# "openai". Anything else LangChain supports will be passed through, but only
# these two are tested.
PROVIDER = os.environ.get("POS_PROVIDER", "azure_openai")

# The model to use. On Azure this is a *deployment* name, chosen by whoever
# created it, so it only matches a catalogue entry when they used the
# model's own name. AZURE_OPENAI_DEPLOYMENT is still honoured so existing
# .env files keep working.
MODEL_NAME = (
    os.environ.get("POS_MODEL")
    or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    or "gpt-5.6-luna"
)

# Models offered in the UI's picker. Azure cannot list deployments with an
# API key alone, so they are named here; leave it unset for just the one
# above.
MODEL_NAMES = [m.strip() for m in os.environ.get("POS_MODELS", "").split(",") if m.strip()] or [MODEL_NAME]

MAX_OUTPUT_TOKENS = _int_env("AZURE_OPENAI_MAX_OUTPUT_TOKENS", 128_000)

# Overrides. Everything below is looked up per model (see models.py); these
# only exist for a deployment the catalogue doesn't know, or a rate that has
# changed before the catalogue caught up. Unset means "use the catalogue".
CONTEXT_WINDOW_OVERRIDE = _int_env("AZURE_OPENAI_CONTEXT_WINDOW", 0) or None
LONG_CONTEXT_THRESHOLD_OVERRIDE = _int_env("AZURE_OPENAI_LONG_CONTEXT_THRESHOLD", 0) or None

PRICE_SHORT_OVERRIDE = {
    "input": _optional_float_env("AZURE_OPENAI_PRICE_INPUT_SHORT"),
    "cached": _optional_float_env("AZURE_OPENAI_PRICE_CACHED_SHORT"),
    "cache_write": _optional_float_env("AZURE_OPENAI_PRICE_CACHE_WRITE_SHORT"),
    "output": _optional_float_env("AZURE_OPENAI_PRICE_OUTPUT_SHORT"),
}
PRICE_LONG_OVERRIDE = {
    "input": _optional_float_env("AZURE_OPENAI_PRICE_INPUT_LONG"),
    "cached": _optional_float_env("AZURE_OPENAI_PRICE_CACHED_LONG"),
    "cache_write": _optional_float_env("AZURE_OPENAI_PRICE_CACHE_WRITE_LONG"),
    "output": _optional_float_env("AZURE_OPENAI_PRICE_OUTPUT_LONG"),
}

# Where the agent works when a request carries no usable folder. Normally
# the UI sends one -- picked through the native dialog -- so this is only
# the fallback for a first run or a bad path.
DEFAULT_ROOT_DIR = os.environ.get("POS_ROOT_DIR") or str(Path.home())
