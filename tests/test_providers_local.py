import os
from unittest.mock import MagicMock

from asr_test.llm.providers.base import ProviderConfig
from asr_test.llm.providers.local import LocalProvider


def _local_provider(model="meta-llama-3.1-8b-instruct"):
    return ProviderConfig(name="local", model=model, base_url="http://localhost:1234/v1", api_key="lm-studio")


def test_detect_is_always_true():
    assert LocalProvider().detect(os.environ) is True


def test_resolve_defaults_when_no_model_override():
    provider = LocalProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "local"
    assert provider.model == "lfm2.5-230m"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.api_key == "lm-studio"


def test_resolve_model_override_wins():
    provider = LocalProvider().resolve(model_override="custom-model", env=os.environ)

    assert provider.model == "custom-model"


def test_resolve_base_url_env_override_wins(monkeypatch):
    monkeypatch.setenv("LOCAL_BASE_URL", "http://192.168.1.50:1234/v1")

    provider = LocalProvider().resolve(model_override=None, env=os.environ)

    assert provider.base_url == "http://192.168.1.50:1234/v1"


def test_resolve_base_url_defaults_when_not_set(monkeypatch):
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)

    provider = LocalProvider().resolve(model_override=None, env=os.environ)

    assert provider.base_url == "http://localhost:1234/v1"


def test_resolve_api_key_env_override_wins(monkeypatch):
    monkeypatch.setenv("LOCAL_API_KEY", "real-lm-studio-token")

    provider = LocalProvider().resolve(model_override=None, env=os.environ)

    assert provider.api_key == "real-lm-studio-token"


def test_resolve_api_key_defaults_when_not_set(monkeypatch):
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)

    provider = LocalProvider().resolve(model_override=None, env=os.environ)

    assert provider.api_key == "lm-studio"


def test_price_for_is_always_free():
    assert LocalProvider().price_for(_local_provider()) == (0.0, 0.0)


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.local.ChatOpenAI", fake_chat_openai)

    result = LocalProvider().build_model(
        _local_provider(model="lfm2.5-230m"), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["base_url"] == "http://localhost:1234/v1"
    assert captured_kwargs["api_key"] == "lm-studio"
    assert captured_kwargs["model"] == "lfm2.5-230m"
    assert captured_kwargs["stream_usage"] is True


def test_context_window_queries_lm_studio_v0_models_endpoint(monkeypatch):
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

    monkeypatch.setattr("asr_test.llm.providers.local.requests.get", fake_get)

    window = LocalProvider().context_window_for(_local_provider())

    assert window == 131072
    assert captured_url["url"] == "http://localhost:1234/api/v0/models"


def test_context_window_prefers_loaded_context_length_over_max_context_length(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "max_context_length": 128000, "loaded_context_length": 8192}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="lfm2.5-230m")) == 8192


def test_context_window_falls_back_to_max_context_length_when_not_loaded(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"id": "lfm2.5-230m", "state": "not-loaded", "max_context_length": 128000}],
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="lfm2.5-230m")) == 128000


def test_context_window_returns_none_when_model_not_found_in_response(monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"object": "list", "data": [{"id": "other-model", "max_context_length": 4096}]}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr("asr_test.llm.providers.local.requests.get", lambda url, timeout: mock_response)

    assert LocalProvider().context_window_for(_local_provider(model="not-listed")) is None


def test_context_window_returns_none_on_request_failure(monkeypatch):
    def fake_get(url, timeout):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("asr_test.llm.providers.local.requests.get", fake_get)

    assert LocalProvider().context_window_for(_local_provider()) is None
