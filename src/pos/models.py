"""What a model costs and how much it can hold.

Neither provider publishes this over an API. Azure's Retail Prices API has
prices but no context window; the `/openai/models` endpoints have
capabilities but neither prices nor limits. models.dev collects both, for
both providers, and its numbers for gpt-5.6-luna were checked against
Microsoft's own Retail Prices API and matched to the cent.

It is community-maintained, so it can lag a price change and may not have a
brand-new deployment at all. Hence: values from `.env` always win, the
answer is cached to disk so a flaky network cannot break startup, and a
model that isn't listed simply has no metadata rather than wrong metadata.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from pos import config, db

logger = logging.getLogger(__name__)

API_URL = "https://models.dev/api.json"
CACHE_PATH = db.DB_DIR / "models.json"
CACHE_TTL_SECONDS = 24 * 60 * 60

# models.dev keys Azure models under "azure", but init_chat_model wants
# "azure_openai". Anthropic and the rest are listed under their own names.
_PROVIDER_ALIASES = {"azure_openai": "azure"}


@dataclasses.dataclass(frozen=True)
class Rates:
    """Per-1M-token USD prices for one context tier.

    Attributes:
        input: Input tokens that were neither cached nor newly written.
        cached: Input tokens served from the prompt cache.
        cache_write: Input tokens newly written to the cache.
        output: Output tokens.
    """

    input: float
    cached: float
    cache_write: float
    output: float


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """Everything known about one model.

    Attributes:
        name: The name to call it by -- a deployment name on Azure.
        context_window: Total tokens it can hold, input plus output.
        max_output: The most it will generate in one reply.
        short: Rates below `long_threshold` input tokens.
        long: Rates above it, or the same as `short` when the model has no
            second tier.
        long_threshold: Where the second tier starts. None when there is
            only one tier.
        reasoning: Whether it accepts a reasoning effort.
    """

    name: str
    context_window: int | None = None
    max_output: int | None = None
    short: Rates | None = None
    long: Rates | None = None
    long_threshold: int | None = None
    reasoning: bool = False


def _fetch() -> dict[str, Any] | None:
    """Downloads the catalogue, or returns None if it can't be reached."""
    # models.dev answers 403 to urllib's default User-Agent, so send a real
    # one naming this app.
    request = urllib.request.Request(API_URL, headers={"User-Agent": "pos-harness"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        logger.warning("could not reach models.dev (%s); using the cached copy", e)
        return None


def catalogue(*, refresh: bool = False) -> dict[str, Any]:
    """The models.dev catalogue, from disk when it's fresh enough.

    Args:
        refresh: Ignore the cache's age and fetch again.

    Returns:
        The catalogue, or `{}` when there is neither a usable cache nor a
        reachable network -- callers treat that as "no metadata".
    """
    cached: dict[str, Any] | None = None
    if CACHE_PATH.is_file():
        age = time.time() - CACHE_PATH.stat().st_mtime
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if cached and not refresh and age < CACHE_TTL_SECONDS:
            return cached

    fresh = _fetch()
    if fresh is None:
        return cached or {}

    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(fresh), encoding="utf-8")
    except OSError as e:
        logger.warning("could not cache the model catalogue: %s", e)
    return fresh


def _candidates(name: str):
    """Names to try, most specific first.

    A reply reports `gpt-5.6-luna-2026-07-09` while the catalogue lists
    `gpt-5.6-luna`, so trailing version segments are peeled off one at a
    time until something matches.

    Args:
        name: The model or deployment name.

    Yields:
        Progressively shorter names.
    """
    parts = name.split("-")
    for stop in range(len(parts), 0, -1):
        yield "-".join(parts[:stop])


def _rates(cost: dict[str, Any] | None) -> Rates | None:
    """Builds a Rates from one of models.dev's cost objects."""
    if not cost:
        return None
    return Rates(
        input=float(cost.get("input") or 0.0),
        cached=float(cost.get("cache_read") or 0.0),
        cache_write=float(cost.get("cache_write") or 0.0),
        output=float(cost.get("output") or 0.0),
    )


def lookup(provider: str, name: str) -> ModelInfo:
    """Finds what a model costs and how much it holds.

    Args:
        provider: "openai", "azure_openai", or any other models.dev
            provider key.
        name: The model name, or on Azure the deployment name -- which only
            matches if whoever created it used the model's own name.

    Returns:
        A ModelInfo. Its fields are None when the model isn't listed; the
        caller falls back to whatever `.env` provides.
    """
    if provider == "lmstudio":
        return lmstudio_info(name)

    data = catalogue()
    key = _PROVIDER_ALIASES.get(provider, provider)
    models = (data.get(key) or {}).get("models") or {}

    entry = None
    for candidate in _candidates(name):
        if candidate in models:
            entry = models[candidate]
            break
    if entry is None:
        return ModelInfo(name=name)

    limit = entry.get("limit") or {}
    cost = entry.get("cost") or {}
    short = _rates(cost)

    # A second tier appears once the prompt passes a size, which is how the
    # gpt-5 family prices long context. No tier means one price throughout.
    long_rates, threshold = short, None
    for tier in cost.get("tiers") or []:
        size = (tier.get("tier") or {}).get("size")
        if size:
            long_rates, threshold = _rates(tier), int(size)
            break

    return ModelInfo(
        name=name,
        context_window=limit.get("context"),
        max_output=limit.get("output"),
        short=short,
        long=long_rates,
        long_threshold=threshold,
        reasoning=bool(entry.get("reasoning")),
    )


OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
_openai_cache: tuple[float, list[str], str | None] | None = None


def openai_models(api_key: str) -> tuple[list[str], str | None]:
    """The chat models an OpenAI key can reach.

    OpenAI, unlike Azure, will list what the key has access to -- so there
    is nothing for the user to type. The raw list also contains embeddings,
    audio and image models, so it is intersected with the catalogue's chat
    entries: that both filters it and guarantees every name returned has
    rates and a context window to show.

    Cached in memory for the process's lifetime plus a TTL, since it barely
    changes and a chat request should not wait on it.

    Args:
        api_key: The OpenAI key.

    Returns:
        Sorted model ids, and a message if the call failed.
    """
    global _openai_cache
    if _openai_cache and time.time() - _openai_cache[0] < CACHE_TTL_SECONDS:
        return _openai_cache[1], _openai_cache[2]

    request = urllib.request.Request(
        OPENAI_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "pos-harness"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as e:
        message = (
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
            if e.code in (401, 403)
            else f"Could not list OpenAI models (HTTP {e.code})."
        )
        _openai_cache = (time.time(), [], message)
        return [], message
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        message = f"Could not reach OpenAI to list models ({e})."
        _openai_cache = (time.time(), [], message)
        return [], message

    known = (catalogue().get("openai") or {}).get("models") or {}
    ids = sorted(m["id"] for m in payload.get("data", []) if m.get("id") in known)
    _openai_cache = (time.time(), ids, None)
    return ids, None


def lmstudio_models(base_url: str) -> tuple[list[str], str | None]:
    """The models an LM Studio server currently has loaded.

    Unlike OpenAI's, these ids won't be in models.dev's catalogue -- they're
    whatever the user downloaded -- so nothing is filtered out. Not cached:
    LM Studio's loaded set changes as the user swaps models, and the call is
    a local, near-instant one anyway.

    Args:
        base_url: LM Studio's OpenAI-compatible base URL, e.g.
            "http://localhost:1234/v1".

    Returns:
        Sorted model ids, and a message if the call failed.
    """
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"User-Agent": "pos-harness"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        return [], f"Could not reach LM Studio at {base_url} ({e})."
    return sorted(m["id"] for m in payload.get("data", [])), None


# A local model costs nothing per token. Real zeros, not _UNKNOWN's
# "we don't know" zeros -- so the model counts as priced.
_FREE = Rates(input=0.0, cached=0.0, cache_write=0.0, output=0.0)


def lmstudio_info(name: str) -> ModelInfo:
    """What LM Studio itself says about one of its models.

    A local model is in no catalogue -- it is whatever file the user
    downloaded -- so the server running it is the only source. Its native
    REST API (`/api/v0/models`, beside the OpenAI-compatible `/v1`) reports
    both the context length the model is loaded with and the most it
    supports; the loaded one is what a conversation actually has to fit in.

    Not cached: the user can reload a model with a different context length
    at any time, and the call is local and near-instant.

    Args:
        name: The model id as LM Studio lists it, e.g. "qwen/qwen3.5-9b".

    Returns:
        A ModelInfo priced at zero. `context_window` is None when LM Studio
        is unreachable or doesn't know the model.
    """
    info = ModelInfo(name=name, short=_FREE, long=_FREE)
    if not config.LMSTUDIO_BASE_URL:
        return info
    # base_url ends in /v1; the native API lives at the same host.
    host = config.LMSTUDIO_BASE_URL.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(f"{host}/api/v0/models", headers={"User-Agent": "pos-harness"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        logger.warning("Could not ask LM Studio about %s: %s", name, e)
        return info

    entry = next((m for m in payload.get("data", []) if m.get("id") == name), None)
    if entry is None:
        return info
    # Not `reasoning`: LM Studio's capabilities list only says "tool_use",
    # and build_model drops reasoning_effort for it regardless.
    return dataclasses.replace(
        info,
        context_window=entry.get("loaded_context_length") or entry.get("max_context_length"),
    )
