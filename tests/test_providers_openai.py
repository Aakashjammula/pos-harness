import os

from asr_test.llm.providers.base import ProviderConfig
from asr_test.llm.providers.openai import OpenAIProvider


def _openai_provider(model="gpt-4o-mini"):
    return ProviderConfig(name="openai", model=model, api_key="sk-test")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert OpenAIProvider().detect(os.environ) is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert OpenAIProvider().detect(os.environ) is False


def test_resolve_uses_default_model_when_no_override_or_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = OpenAIProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"
    assert provider.base_url is None


def test_resolve_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = OpenAIProvider().resolve(model_override=None, env=os.environ)

    assert provider.model == "gpt-4o"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = OpenAIProvider().resolve(model_override="gpt-4o-mini", env=os.environ)

    assert provider.model == "gpt-4o-mini"


def test_price_for_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert OpenAIProvider().price_for(_openai_provider()) == (0.15, 0.60)


def test_price_for_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert OpenAIProvider().price_for(_openai_provider(model="some-unlisted-model")) is None


def test_price_for_env_override_wins_over_built_in_table(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    assert OpenAIProvider().price_for(_openai_provider()) == (1.0, 2.0)


def test_context_window_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)

    assert OpenAIProvider().context_window_for(_openai_provider()) == 128_000


def test_context_window_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)

    assert OpenAIProvider().context_window_for(_openai_provider(model="some-unlisted-model")) is None


def test_context_window_env_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "200000")

    assert OpenAIProvider().context_window_for(_openai_provider()) == 200000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.openai.ChatOpenAI", fake_chat_openai)

    result = OpenAIProvider().build_model(
        _openai_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["base_url"] is None
    assert captured_kwargs["api_key"] == "sk-test"
    assert captured_kwargs["model"] == "gpt-4o-mini"
