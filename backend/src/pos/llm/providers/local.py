"""LM Studio (or any OpenAI-compatible local server). Detected only when a
base URL is given (LOCAL_BASE_URL) -- nothing is assumed, so with no URL and
no other provider configured there is no LLM at all. Checked last in
resolve_provider()'s order (highest priority number). See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import requests
from langchain_openai import ChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_PLACEHOLDER_API_KEY = "lm-studio"   # the OpenAI SDK insists on a non-empty key; LM Studio
                                     # ignores it unless its own "Require Authentication"
                                     # setting is on, in which case LOCAL_API_KEY must carry
                                     # its real token. Not a configuration default.


def _first_served_model(base_url: str, api_key: str) -> str | None:
    """The first model id the server reports, or None. Used only when the
    user gave a URL but did not name a model."""
    try:
        resp = requests.get(
            f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=3
        )
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        return ids[0] if ids else None
    except Exception:
        return None


@register
class LocalProvider(LlmProviderBase):
    name = "local"
    priority = 100   # checked last

    def detect(self, env) -> bool:
        return bool(env.get("LOCAL_BASE_URL"))

    def resolve(self, model_override: str | None, env) -> ProviderConfig:
        base_url = env.get("LOCAL_BASE_URL")
        if not base_url:
            raise RuntimeError("LOCAL_BASE_URL is not set")
        api_key = env.get("LOCAL_API_KEY") or _PLACEHOLDER_API_KEY
        model = model_override or env.get("LOCAL_MODEL") or _first_served_model(base_url, api_key)
        if not model:
            raise RuntimeError(
                f"no model chosen and none reported by {base_url} -- load a model there or set LOCAL_MODEL"
            )
        return ProviderConfig(name=self.name, model=model, base_url=base_url, api_key=api_key)

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return ChatOpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            model=provider.model,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        return (0.0, 0.0)

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        host = (provider.base_url or "").removesuffix("/v1")
        try:
            resp = requests.get(f"{host}/api/v0/models", timeout=3)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == provider.model:
                    # loaded_context_length is what LM Studio actually
                    # configured for the running instance (e.g. 8192, set
                    # in its UI) -- max_context_length is the model's
                    # architectural ceiling (e.g. 128000) and can be much
                    # larger than what's really available. Prefer the
                    # real, currently-in-effect value; only fall back to
                    # the ceiling if the model isn't loaded
                    # (loaded_context_length absent/None in that state).
                    return entry.get("loaded_context_length") or entry.get("max_context_length")
        except Exception:
            return None
        return None
