"""Resolves which LLM backend to talk to from environment variables —
env-var-only, decided once per process, no per-session picker (see
docs/superpowers/specs/2026-09-10-llm-provider-cost-tracking-design.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_LOCAL_MODEL = "lfm2.5-230m"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_AZURE_API_VERSION = "2026-01-01-preview"


@dataclass
class ProviderConfig:
    name: str                        # "local" | "openai" | "azure"
    model: str
    base_url: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None


def resolve_provider(model_override: str | None = None) -> ProviderConfig:
    """Precedence: AZURE_OPENAI_API_KEY set -> azure; else OPENAI_API_KEY
    set -> openai; else -> local. Raises RuntimeError naming the missing
    var if a selected backend's other required vars aren't set — fails
    at construction time, never silently falls back to local."""

    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is required when AZURE_OPENAI_API_KEY is set"
            )
        deployment = model_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError(
                "AZURE_OPENAI_DEPLOYMENT is required when AZURE_OPENAI_API_KEY is "
                "set (or pass model=... to override it)"
            )
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION)
        return ProviderConfig(
            name="azure",
            model=deployment,
            api_key=azure_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model = model_override or os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        return ProviderConfig(name="openai", model=model, api_key=openai_key)

    return ProviderConfig(
        name="local",
        model=model_override or _DEFAULT_LOCAL_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
