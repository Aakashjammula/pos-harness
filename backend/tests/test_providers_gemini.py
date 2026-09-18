import os

from pos.llm.providers.base import ProviderConfig
from pos.llm.providers.gemini import GeminiProvider


def _gemini_provider(model="gemini-3.7-flash"):
    return ProviderConfig(name="gemini", model=model, api_key="fake-google-key")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")

    assert GeminiProvider().detect(os.environ) is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert GeminiProvider().detect(os.environ) is False


def test_resolve_uses_default_model_when_no_override_or_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    provider = GeminiProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "gemini"
    assert provider.api_key == "fake-google-key"
    assert provider.model == "gemini-3.7-flash"


def test_resolve_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.7-pro")

    provider = GeminiProvider().resolve(model_override=None, env=os.environ)

    assert provider.model == "gemini-3.7-pro"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.7-pro")

    provider = GeminiProvider().resolve(model_override="gemini-3.7-flash", env=os.environ)

    assert provider.model == "gemini-3.7-flash"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("GEMINI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("GEMINI_PRICE_OUTPUT_PER_1K", raising=False)

    assert GeminiProvider().price_for(_gemini_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("GEMINI_PRICE_INPUT_PER_1K", "0.075")
    monkeypatch.setenv("GEMINI_PRICE_OUTPUT_PER_1K", "0.3")

    assert GeminiProvider().price_for(_gemini_provider()) == (0.075, 0.3)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("GEMINI_CONTEXT_WINDOW", raising=False)

    assert GeminiProvider().context_window_for(_gemini_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("GEMINI_CONTEXT_WINDOW", "1000000")

    assert GeminiProvider().context_window_for(_gemini_provider()) == 1000000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_gemini(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("pos.llm.providers.gemini.ChatGoogleGenerativeAI", fake_chat_gemini)

    result = GeminiProvider().build_model(
        _gemini_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["model"] == "gemini-3.7-flash"
    assert captured_kwargs["api_key"] == "fake-google-key"
    assert captured_kwargs["max_tokens"] == 120


def test_build_model_strips_unsupported_stream_usage_kwarg(monkeypatch):
    # ChatGoogleGenerativeAI doesn't accept stream_usage -- passing it
    # through would emit a UserWarning and silently stuff it into
    # model_kwargs instead of doing anything useful.
    captured_kwargs = {}

    def fake_chat_gemini(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("pos.llm.providers.gemini.ChatGoogleGenerativeAI", fake_chat_gemini)

    GeminiProvider().build_model(_gemini_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True)

    assert "stream_usage" not in captured_kwargs
