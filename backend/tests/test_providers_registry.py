import pytest

from pos.llm.providers import ProviderConfig, estimate_cost, is_configured, resolve_provider


_ALL_PROVIDER_ENV = (
    "LOCAL_BASE_URL", "LOCAL_MODEL", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "AWS_ACCESS_KEY_ID", "OPENROUTER_API_KEY",
)


def _clear_all(monkeypatch):
    for name in _ALL_PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def test_no_provider_is_assumed_when_nothing_is_configured(monkeypatch):
    _clear_all(monkeypatch)

    assert is_configured() is False
    with pytest.raises(RuntimeError, match="no LLM provider configured"):
        resolve_provider()


def test_local_is_selected_only_once_a_base_url_is_given(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("LOCAL_BASE_URL", "http://my-host:1234/v1")
    monkeypatch.setenv("LOCAL_MODEL", "my-model")

    provider = resolve_provider()

    assert is_configured() is True
    assert provider.name == "local"
    assert provider.base_url == "http://my-host:1234/v1"
    assert provider.model == "my-model"


def test_openai_backend_selected_when_key_set(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = resolve_provider()

    assert provider.name == "openai"
    assert provider.api_key == "sk-test"
    assert provider.model == "gpt-4o-mini"


def test_azure_backend_takes_precedence_over_openai(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # both set — azure must win

    provider = resolve_provider()

    assert provider.name == "azure"
    assert provider.model == "my-deployment"


def test_model_override_wins_regardless_of_which_provider_is_selected(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = resolve_provider(model_override="gpt-4o")

    assert provider.model == "gpt-4o"


def test_azure_missing_endpoint_raises(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        resolve_provider()


def test_estimate_cost_computes_weighted_sum(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    # 2000 input tokens @ $1.0/1K = $2.0; 500 output tokens @ $2.0/1K = $1.0
    cost = estimate_cost(provider, input_tokens=2000, output_tokens=500)

    assert cost == 3.0


def test_estimate_cost_local_always_zero():
    provider = ProviderConfig(name="local", model="anything", base_url="http://localhost:1234/v1", api_key="lm-studio")

    assert estimate_cost(provider, input_tokens=999999, output_tokens=999999) == 0.0


def test_estimate_cost_returns_none_when_unpriced(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)
    provider = ProviderConfig(name="openai", model="some-unlisted-model", api_key="sk-test")

    assert estimate_cost(provider, input_tokens=1000, output_tokens=1000) is None
