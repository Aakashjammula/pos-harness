import os

from asr_test.llm.providers.base import ProviderConfig
from asr_test.llm.providers.openrouter import OpenRouterProvider


def _openrouter_provider(model="openrouter/auto"):
    return ProviderConfig(name="openrouter", model=model, api_key="sk-or-test")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert OpenRouterProvider().detect(os.environ) is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert OpenRouterProvider().detect(os.environ) is False


def test_resolve_uses_default_model_when_no_override_or_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    provider = OpenRouterProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "openrouter"
    assert provider.api_key == "sk-or-test"
    assert provider.model == "openrouter/auto"


def test_resolve_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")

    provider = OpenRouterProvider().resolve(model_override=None, env=os.environ)

    assert provider.model == "anthropic/claude-sonnet-4-6"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")

    provider = OpenRouterProvider().resolve(model_override="openai/gpt-4o-mini", env=os.environ)

    assert provider.model == "openai/gpt-4o-mini"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENROUTER_PRICE_OUTPUT_PER_1K", raising=False)

    assert OpenRouterProvider().price_for(_openrouter_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENROUTER_PRICE_OUTPUT_PER_1K", "2.0")

    assert OpenRouterProvider().price_for(_openrouter_provider()) == (1.0, 2.0)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_CONTEXT_WINDOW", raising=False)

    assert OpenRouterProvider().context_window_for(_openrouter_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("OPENROUTER_CONTEXT_WINDOW", "128000")

    assert OpenRouterProvider().context_window_for(_openrouter_provider()) == 128000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openrouter(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.openrouter.ChatOpenRouter", fake_chat_openrouter)

    result = OpenRouterProvider().build_model(
        _openrouter_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["model"] == "openrouter/auto"
    assert captured_kwargs["api_key"] == "sk-or-test"
    assert captured_kwargs["stream_usage"] is True
