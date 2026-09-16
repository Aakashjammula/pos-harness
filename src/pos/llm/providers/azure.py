"""Azure OpenAI. See
docs/superpowers/specs/2026-09-10-llm-provider-registry-and-src-layout-design.md."""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_API_VERSION = "2026-01-01-preview"


@register
class AzureProvider(LlmProviderBase):
    name = "azure"
    priority = 0   # checked first

    def detect(self, env) -> bool:
        return bool(env.get("AZURE_OPENAI_API_KEY"))

    def resolve(self, model_override: str | None, env) -> ProviderConfig:
        api_key = env["AZURE_OPENAI_API_KEY"]
        endpoint = env.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is required when AZURE_OPENAI_API_KEY is set"
            )
        deployment = model_override or env.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError(
                "AZURE_OPENAI_DEPLOYMENT is required when AZURE_OPENAI_API_KEY is "
                "set (or pass model=... to override it)"
            )
        api_version = env.get("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION)
        return ProviderConfig(
            name=self.name,
            model=deployment,
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
        )

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        return AzureChatOpenAI(
            azure_endpoint=provider.azure_endpoint,
            azure_deployment=provider.azure_deployment,
            api_version=provider.api_version,
            api_key=provider.api_key,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("AZURE_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("AZURE_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return None

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("AZURE_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return None
