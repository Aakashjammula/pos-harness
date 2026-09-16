"""Public surface for the LLM provider registry. Importing this module
imports azure.py/local.py/openai.py/anthropic.py/gemini.py/bedrock.py/
openrouter.py purely for their @register side effect -- each self-registers
into registry.py's _REGISTRY when its module loads. Adding a new provider
means adding one file and one import line here; resolve_provider() and
friends never need to change."""

from .anthropic import AnthropicProvider
from .azure import AzureProvider
from .base import ProviderConfig
from .bedrock import BedrockProvider
from .gemini import GeminiProvider
from .local import LocalProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider
from .registry import build_model, context_window_for, estimate_cost, price_for, resolve_provider

__all__ = [
    "ProviderConfig",
    "resolve_provider",
    "build_model",
    "price_for",
    "context_window_for",
    "estimate_cost",
    "AnthropicProvider",
    "AzureProvider",
    "BedrockProvider",
    "GeminiProvider",
    "LocalProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
