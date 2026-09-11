"""LM Studio (or any OpenAI-compatible local server). Always detected
(the fallback provider -- lowest priority in resolve_provider()'s
checking order, i.e. the highest priority number). See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import requests
from langchain_openai import ChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "lfm2.5-230m"
_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_API_KEY = "lm-studio"   # a harmless placeholder -- LM Studio ignores it
                                 # unless its own "Require Authentication" setting
                                 # is turned on, in which case LOCAL_API_KEY below
                                 # must carry its real token.


@register
class LocalProvider(LlmProviderBase):
    name = "local"
    priority = 100   # fallback -- always matches, checked last

    def detect(self, env) -> bool:
        return True

    def resolve(self, model_override: str | None, env) -> ProviderConfig:
        return ProviderConfig(
            name=self.name,
            model=model_override or _DEFAULT_MODEL,
            base_url=env.get("LOCAL_BASE_URL", _DEFAULT_BASE_URL),
            api_key=env.get("LOCAL_API_KEY", _DEFAULT_API_KEY),
        )

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
