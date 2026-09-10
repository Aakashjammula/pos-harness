"""Context-window size lookup, paired with pricing.py's tables. Local
(LM Studio) exposes this live via its REST API v0 -- see
https://lmstudio.ai/docs/developer/rest/endpoints -- a DIFFERENT base
path (/api/v0/...) than the OpenAI-compatible /v1/models this project
calls elsewhere for the model-name dropdown. OpenAI's API has no
endpoint that returns a model's context window at all (confirmed via
research, not an oversight); Azure doesn't either, since it also
depends on which base model was deployed. Both fall back to a small
built-in table, same env-override pattern as pricing.py."""

from __future__ import annotations

import os

import requests

from .provider import ProviderConfig

CONTEXT_WINDOWS: dict[tuple[str, str], int] = {
    # (provider, model) -> max context window, in tokens.
    # Best-effort snapshot, same caveat as pricing.py's PRICING table --
    # override via {PROVIDER}_CONTEXT_WINDOW if this has changed or a
    # new model needs one.
    ("openai", "gpt-4o-mini"): 128_000,
    ("openai", "gpt-4o"): 128_000,
}


def _local_context_window(base_url: str | None, model: str) -> int | None:
    host = (base_url or "").removesuffix("/v1")
    try:
        resp = requests.get(f"{host}/api/v0/models", timeout=3)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("id") == model:
                # loaded_context_length is what LM Studio actually configured
                # for the running instance (e.g. 8192, set in its UI) --
                # max_context_length is the model's architectural ceiling
                # (e.g. 128000) and can be much larger than what's really
                # available. Prefer the real, currently-in-effect value;
                # only fall back to the ceiling if the model isn't loaded
                # (loaded_context_length is absent/None in that state).
                return entry.get("loaded_context_length") or entry.get("max_context_length")
    except Exception:
        return None
    return None


def get_context_window(provider: ProviderConfig) -> int | None:
    if provider.name == "local":
        return _local_context_window(provider.base_url, provider.model)

    env_val = os.environ.get(f"{provider.name.upper()}_CONTEXT_WINDOW")
    if env_val is not None:
        return int(env_val)

    return CONTEXT_WINDOWS.get((provider.name, provider.model))
