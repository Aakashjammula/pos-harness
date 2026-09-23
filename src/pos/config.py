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


def _detect_provider() -> tuple[str, str | None]:
    """Works out which provider to use from the keys that are present.

    Azure is chosen when it has both the endpoint and the key it needs,
    then OpenAI when its key is present. Azure is checked first only
    because it needs two variables, so its presence is the more deliberate
    signal -- if both are set, delete the one you don't want.

    Returns:
        The provider name, and a description of what's missing -- None when
        the configuration is usable.
    """
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure_openai", None
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", None

    # Half-configured Azure is the likeliest mistake, so name the missing half.
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure_openai", "AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY is not. Add it to .env."
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure_openai", "AZURE_OPENAI_API_KEY is set but AZURE_OPENAI_ENDPOINT is not. Add it to .env."

    return "azure_openai", (
        "No provider configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, "
        "or OPENAI_API_KEY, in .env -- see .env.example."
    )


# The provider used when a model name doesn't name one itself. Follows from
# which keys exist; there is nothing to configure.
PROVIDER, CONFIG_ERROR = _detect_provider()

# Which providers this .env can actually reach.
AVAILABLE_PROVIDERS = {
    name
    for name, ready in (
        ("azure_openai", bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))),
        ("openai", bool(os.environ.get("OPENAI_API_KEY"))),
    )
    if ready
}

KEY_VARIABLES = {"azure_openai": "AZURE_OPENAI_API_KEY", "openai": "OPENAI_API_KEY"}


def split_model(spec: str) -> tuple[str, str]:
    """Splits a model specification into its provider and model name.

    `init_chat_model` already understands "<provider>:<model>", so naming
    the provider in the model is the cheapest way to let one .env hold keys
    for both and switch between them from the picker. A bare name means
    whichever provider the keys point at.

    Args:
        spec: "openai:gpt-5", "azure_openai:my-deployment", or "gpt-5".

    Returns:
        (provider, model name).
    """
    provider, separator, name = spec.partition(":")
    if separator and provider in KEY_VARIABLES:
        return provider, name
    return PROVIDER, spec

# The model to use. On Azure this is a *deployment* name, chosen by whoever
# created it, so it only matches a catalogue entry when they used the
# model's own name. AZURE_OPENAI_DEPLOYMENT is still honoured so existing
# .env files keep working.
#
# Empty when nothing is set. There is deliberately no default: a name baked
# in here would be a deployment that exists on one Azure resource and
# nowhere else, so an unset POS_MODEL would fail with a 404 for a name the
# user never typed. With an OpenAI key the list can be discovered instead
# (see app.list_models), and Azure is told to set it.
MODEL_NAME = os.environ.get("POS_MODEL") or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or ""

# Models offered in the UI's picker, each optionally prefixed with its
# provider ("openai:gpt-5"). Azure cannot list deployments with an API key
# alone, so they are named here; leave it unset for just the one above.
MODEL_SPECS = [m.strip() for m in os.environ.get("POS_MODELS", "").split(",") if m.strip()] or (
    [MODEL_NAME] if MODEL_NAME else []
)

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


def azure_specs() -> list[str]:
    """The Azure models to offer -- the ones named in .env.

    Azure cannot list a resource's deployments from an API key, and a
    deployment's name is whatever its creator chose, so there is nothing to
    discover. OpenAI models are not included here; they are listed from the
    key itself (see models.openai_models).

    Returns:
        Specs whose provider is Azure, with the prefix left as written.
    """
    return [spec for spec in MODEL_SPECS if split_model(spec)[0] == "azure_openai"]


def model_error(spec: str) -> str | None:
    """Why this model can't be called, or None when it can.

    A model may name a provider whose keys are absent -- POS_MODELS can
    list both providers while .env only has one set of keys -- or there may
    be no model at all, which is what an empty POS_MODEL on Azure means.

    Args:
        spec: The model as configured.

    Returns:
        A sentence naming what to set, or None.
    """
    if not spec:
        return (
            "No model configured. Set POS_MODEL in .env -- an Azure deployment name cannot be "
            "discovered from an API key, so there is nothing to fall back to."
        )
    provider, _ = split_model(spec)
    if provider in AVAILABLE_PROVIDERS:
        return None
    if not AVAILABLE_PROVIDERS:
        return CONFIG_ERROR
    return f"{spec} needs {KEY_VARIABLES[provider]} in .env."
