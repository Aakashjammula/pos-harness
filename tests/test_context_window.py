from unittest.mock import MagicMock

from asr_test.llm.context_window import get_context_window
from asr_test.llm.provider import ProviderConfig


def _local_provider(model="meta-llama-3.1-8b-instruct"):
    return ProviderConfig(
        name="local", model=model,
        base_url="http://localhost:1234/v1", api_key="lm-studio",
    )


def test_local_queries_lm_studio_v0_models_endpoint(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {"id": "qwen2-vl-7b-instruct", "max_context_length": 32768},
            {"id": "meta-llama-3.1-8b-instruct", "max_context_length": 131072},
        ],
    }
    mock_response.raise_for_status.return_value = None
    captured_url = {}

    def fake_get(url, timeout):
        captured_url["url"] = url
        return mock_response

    monkeypatch.setattr("asr_test.llm.context_window.requests.get", fake_get)

    window = get_context_window(_local_provider())

    assert window == 131072
    assert captured_url["url"] == "http://localhost:1234/api/v0/models"


def test_local_prefers_loaded_context_length_over_max_context_length(monkeypatch):
    # Real-world case: a model with a 128000-token architectural max can be
    # loaded with a much smaller configured context (e.g. 8192 in LM
    # Studio's own UI) -- loaded_context_length reflects what's actually
    # in effect, max_context_length is just the ceiling.
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "max_context_length": 128000, "loaded_context_length": 8192}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.context_window.requests.get", lambda url, timeout: mock_response)

    assert get_context_window(_local_provider(model="lfm2.5-230m")) == 8192


def test_local_falls_back_to_max_context_length_when_not_loaded(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "state": "not-loaded", "max_context_length": 128000}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.context_window.requests.get", lambda url, timeout: mock_response)

    assert get_context_window(_local_provider(model="lfm2.5-230m")) == 128000


def test_local_returns_none_when_model_not_found_in_response(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"object": "list", "data": [{"id": "other-model", "max_context_length": 4096}]}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.context_window.requests.get", lambda url, timeout: mock_response)

    assert get_context_window(_local_provider(model="not-listed")) is None


def test_local_returns_none_on_request_failure(monkeypatch):
    def fake_get(url, timeout):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("asr_test.llm.context_window.requests.get", fake_get)

    assert get_context_window(_local_provider()) is None


def test_openai_known_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    assert get_context_window(provider) == 128_000


def test_openai_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="openai", model="some-unlisted-model", api_key="sk-test")

    assert get_context_window(provider) is None


def test_azure_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_CONTEXT_WINDOW", raising=False)
    provider = ProviderConfig(name="azure", model="my-deployment", api_key="azure-key")

    assert get_context_window(provider) is None


def test_env_override_wins_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "200000")
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key="sk-test")

    assert get_context_window(provider) == 200000


def test_env_override_enables_azure(monkeypatch):
    monkeypatch.setenv("AZURE_CONTEXT_WINDOW", "128000")
    provider = ProviderConfig(name="azure", model="my-deployment", api_key="azure-key")

    assert get_context_window(provider) == 128000
