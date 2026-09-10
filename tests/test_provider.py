import pytest

from asr_test.llm.provider import resolve_provider


def test_defaults_to_local_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = resolve_provider()

    assert provider.name == "local"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key == "lm-studio"
    assert provider.model == "lfm2.5-230m"


def test_model_override_wins_for_local(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = resolve_provider(model_override="custom-model")

    assert provider.name == "local"
    assert provider.model == "custom-model"


def test_openai_backend_selected_when_key_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = resolve_provider()

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"
    assert provider.base_url is None


def test_openai_model_env_var_used_when_no_override(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = resolve_provider()

    assert provider.model == "gpt-4o"


def test_model_override_wins_over_openai_model_env_var(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    provider = resolve_provider(model_override="gpt-4o-mini")

    assert provider.model == "gpt-4o-mini"


def test_azure_backend_takes_precedence_over_openai(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # both set — azure must win

    provider = resolve_provider()

    assert provider.name == "azure"
    assert provider.api_key == "azure-key"
    assert provider.azure_endpoint == "https://example.openai.azure.com/"
    assert provider.azure_deployment == "my-deployment"
    assert provider.model == "my-deployment"
    assert provider.api_version == "2026-01-01-preview"


def test_azure_api_version_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-05-01")

    provider = resolve_provider()

    assert provider.api_version == "2025-05-01"


def test_azure_model_override_wins_over_deployment_env_var(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    provider = resolve_provider(model_override="other-deployment")

    assert provider.model == "other-deployment"
    assert provider.azure_deployment == "other-deployment"


def test_azure_missing_endpoint_raises(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        resolve_provider()


def test_azure_missing_deployment_raises_when_no_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"):
        resolve_provider()
