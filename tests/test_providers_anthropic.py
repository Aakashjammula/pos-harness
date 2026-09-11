import os

from asr_test.llm.providers.anthropic import AnthropicProvider
from asr_test.llm.providers.base import ProviderConfig


def _anthropic_provider(model="claude-sonnet-4-6"):
    return ProviderConfig(name="anthropic", model=model, api_key="sk-ant-test")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert AnthropicProvider().detect(os.environ) is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert AnthropicProvider().detect(os.environ) is False


def test_resolve_uses_default_model_when_no_override_or_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    provider = AnthropicProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "anthropic"
    assert provider.api_key == "sk-ant-test"
    assert provider.model == "claude-sonnet-4-6"


def test_resolve_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-6")

    provider = AnthropicProvider().resolve(model_override=None, env=os.environ)

    assert provider.model == "claude-opus-4-6"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-6")

    provider = AnthropicProvider().resolve(model_override="claude-haiku-4-6", env=os.environ)

    assert provider.model == "claude-haiku-4-6"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("ANTHROPIC_PRICE_OUTPUT_PER_1K", raising=False)

    assert AnthropicProvider().price_for(_anthropic_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("ANTHROPIC_PRICE_OUTPUT_PER_1K", "15.0")

    assert AnthropicProvider().price_for(_anthropic_provider()) == (3.0, 15.0)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_CONTEXT_WINDOW", raising=False)

    assert AnthropicProvider().context_window_for(_anthropic_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CONTEXT_WINDOW", "200000")

    assert AnthropicProvider().context_window_for(_anthropic_provider()) == 200000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_anthropic(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.anthropic.ChatAnthropic", fake_chat_anthropic)

    result = AnthropicProvider().build_model(
        _anthropic_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["model"] == "claude-sonnet-4-6"
    assert captured_kwargs["api_key"] == "sk-ant-test"
    assert captured_kwargs["stream_usage"] is True
