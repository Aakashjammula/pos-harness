import pytest

from asr_test.llm.providers.azure import AzureProvider
from asr_test.llm.providers.base import ProviderConfig


def _azure_provider(model="my-deployment"):
    return ProviderConfig(name="azure", model=model, api_key="azure-key")


def test_detect_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")

    assert AzureProvider().detect() is True


def test_detect_false_when_api_key_not_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    assert AzureProvider().detect() is False


def test_resolve_raises_when_endpoint_missing(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        AzureProvider().resolve(model_override=None)


def test_resolve_raises_when_deployment_missing_and_no_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"):
        AzureProvider().resolve(model_override=None)


def test_resolve_model_override_wins_over_deployment_env_var(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    provider = AzureProvider().resolve(model_override="other-deployment")

    assert provider.model == "other-deployment"
    assert provider.azure_deployment == "other-deployment"


def test_resolve_api_version_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-05-01")

    provider = AzureProvider().resolve(model_override=None)

    assert provider.api_version == "2025-05-01"


def test_resolve_api_version_defaults_when_not_set(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    provider = AzureProvider().resolve(model_override=None)

    assert provider.api_version == "2026-01-01-preview"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("AZURE_PRICE_OUTPUT_PER_1K", raising=False)

    assert AzureProvider().price_for(_azure_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("AZURE_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("AZURE_PRICE_OUTPUT_PER_1K", "4.0")

    assert AzureProvider().price_for(_azure_provider()) == (3.0, 4.0)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_CONTEXT_WINDOW", raising=False)

    assert AzureProvider().context_window_for(_azure_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("AZURE_CONTEXT_WINDOW", "128000")

    assert AzureProvider().context_window_for(_azure_provider()) == 128000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_azure_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.azure.AzureChatOpenAI", fake_azure_chat_openai)
    provider = ProviderConfig(
        name="azure", model="my-deployment", api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/", azure_deployment="my-deployment",
        api_version="2026-01-01-preview",
    )

    result = AzureProvider().build_model(provider, max_tokens=120, temperature=0.7, timeout=30, stream_usage=True)

    assert result == "the-model"
    assert captured_kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert captured_kwargs["azure_deployment"] == "my-deployment"
    assert captured_kwargs["api_key"] == "azure-key"
    assert captured_kwargs["api_version"] == "2026-01-01-preview"
