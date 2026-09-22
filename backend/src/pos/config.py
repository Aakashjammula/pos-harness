"""Settings read from the environment.

Single local user, no database of credentials: every setting here comes
from `.env`, edited by hand. See `.env.example` for the full list.
"""

from __future__ import annotations

import dataclasses
import os

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


# The one Azure OpenAI deployment this server talks to. Azure's "deployment
# name" is what init_chat_model needs here; it's often, but not always, the
# same string as the underlying model name.
MODEL_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-luna")

CONTEXT_WINDOW = _int_env("AZURE_OPENAI_CONTEXT_WINDOW", 1_050_000)
MAX_OUTPUT_TOKENS = _int_env("AZURE_OPENAI_MAX_OUTPUT_TOKENS", 128_000)
LONG_CONTEXT_THRESHOLD = _int_env("AZURE_OPENAI_LONG_CONTEXT_THRESHOLD", 272_000)

PRICE_SHORT = PriceTier(
    input=_float_env("AZURE_OPENAI_PRICE_INPUT_SHORT", 0.20),
    cached=_float_env("AZURE_OPENAI_PRICE_CACHED_SHORT", 0.02),
    cache_write=_float_env("AZURE_OPENAI_PRICE_CACHE_WRITE_SHORT", 0.25),
    output=_float_env("AZURE_OPENAI_PRICE_OUTPUT_SHORT", 1.20),
)
PRICE_LONG = PriceTier(
    input=_float_env("AZURE_OPENAI_PRICE_INPUT_LONG", 0.40),
    cached=_float_env("AZURE_OPENAI_PRICE_CACHED_LONG", 0.04),
    cache_write=_float_env("AZURE_OPENAI_PRICE_CACHE_WRITE_LONG", 0.50),
    output=_float_env("AZURE_OPENAI_PRICE_OUTPUT_LONG", 1.80),
)

# TODO: hardcoded until the folder-picker -> real-path problem is resolved
# (the browser's File System Access API cannot hand the backend a real
# filesystem path). See the backend design discussion.
DEFAULT_ROOT_DIR = os.environ.get("POS_ROOT_DIR", r"C:\Users\VH0000543\Downloads\docs")
