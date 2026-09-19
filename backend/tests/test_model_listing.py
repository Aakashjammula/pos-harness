import pytest
import requests

from pos.llm import model_listing
from pos.llm.model_listing import ModelListError, list_models


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def http(monkeypatch):
    """Record every requests.get and answer from a per-URL queue."""
    calls, queue = [], {}

    def fake_get(url, headers=None, params=None, timeout=None, **kw):
        calls.append({"url": url, "headers": headers or {}, "params": params or {}})
        return queue[url].pop(0)

    monkeypatch.setattr("pos.llm.model_listing.requests.get", fake_get)
    return type("H", (), {"calls": calls, "queue": queue})


def test_openai_drops_non_chat_models_and_sorts_newest_first(http):
    http.queue["https://api.openai.com/v1/models"] = [_Resp({"data": [
        {"id": "gpt-old", "created": 1},
        {"id": "text-embedding-3-small", "created": 9},
        {"id": "whisper-1", "created": 9},
        {"id": "gpt-new", "created": 5},
        {"id": "gpt-4o-audio-preview", "created": 7},
    ]})]

    models = list_models("openai", {"OPENAI_API_KEY": "sk-secret"})

    assert [m["id"] for m in models] == ["gpt-new", "gpt-old"]
    assert http.calls[0]["headers"]["Authorization"] == "Bearer sk-secret"


def test_anthropic_follows_pagination_and_uses_display_names(http):
    url = "https://api.anthropic.com/v1/models"
    http.queue[url] = [
        _Resp({"data": [{"id": "m1", "display_name": "Model One"}], "has_more": True, "last_id": "m1"}),
        _Resp({"data": [{"id": "m2"}], "has_more": False, "last_id": "m2"}),
    ]

    models = list_models("anthropic", {"ANTHROPIC_API_KEY": "k"})

    assert models == [{"id": "m1", "label": "Model One"}, {"id": "m2", "label": "m2"}]
    assert http.calls[1]["params"]["after_id"] == "m1"
    assert http.calls[0]["headers"] == {"x-api-key": "k", "anthropic-version": "2023-06-01"}


def test_gemini_keeps_only_generate_content_models_and_strips_the_prefix(http):
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    http.queue[url] = [
        _Resp({"models": [
            {"name": "models/gemini-a", "displayName": "Gemini A", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed-x", "supportedGenerationMethods": ["embedContent"]},
        ], "nextPageToken": "t2"}),
        _Resp({"models": [
            {"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        ]}),
    ]

    models = list_models("gemini", {"GOOGLE_API_KEY": "g-secret"})

    assert models == [{"id": "gemini-a", "label": "Gemini A"}, {"id": "gemini-b", "label": "gemini-b"}]
    assert http.calls[1]["params"]["pageToken"] == "t2"
    # the key travels in a header, never in the URL or query string
    assert all("g-secret" not in str(c["url"]) + str(c["params"]) for c in http.calls)
    assert http.calls[0]["headers"]["x-goog-api-key"] == "g-secret"


def test_openrouter_keeps_text_models_that_support_tools(http):
    http.queue["https://openrouter.ai/api/v1/models"] = [_Resp({"data": [
        {"id": "a/chat", "name": "Chat", "architecture": {"output_modalities": ["text"]},
         "supported_parameters": ["tools"]},
        {"id": "b/image", "architecture": {"output_modalities": ["image"]}, "supported_parameters": ["tools"]},
        {"id": "c/no-tools", "architecture": {"output_modalities": ["text"]}, "supported_parameters": ["temperature"]},
        {"id": "d/unknown-fields"},
    ]})]

    assert [m["id"] for m in list_models("openrouter", {"OPENROUTER_API_KEY": "k"})] == ["a/chat", "d/unknown-fields"]


def test_local_lists_the_servers_models_from_the_given_url(http):
    http.queue["http://my-host:1234/v1/models"] = [_Resp({"data": [{"id": "qwen"}, {"id": "llama"}]})]

    models = list_models("local", {"LOCAL_BASE_URL": "http://my-host:1234/v1/"})

    assert [m["id"] for m in models] == ["qwen", "llama"]


def test_a_missing_credential_field_is_a_clear_error():
    with pytest.raises(ModelListError, match="no API key"):
        list_models("openai", {})


def test_azure_cannot_be_listed():
    with pytest.raises(ModelListError, match="deployment name"):
        list_models("azure", {"AZURE_OPENAI_API_KEY": "k"})


def test_unknown_provider_is_rejected():
    with pytest.raises(ModelListError):
        list_models("tavily", {})


@pytest.mark.parametrize("status,fragment", [(401, "rejected this key"), (403, "rejected this key"), (500, "HTTP 500")])
def test_http_errors_become_readable_messages(http, status, fragment):
    http.queue["https://api.openai.com/v1/models"] = [_Resp({}, status)]

    with pytest.raises(ModelListError, match=fragment):
        list_models("openai", {"OPENAI_API_KEY": "sk-secret"})


def test_a_network_failure_never_leaks_the_key(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError(f"failed for {url}?key=sk-secret")

    monkeypatch.setattr("pos.llm.model_listing.requests.get", boom)

    with pytest.raises(ModelListError) as info:
        list_models("gemini", {"GOOGLE_API_KEY": "sk-secret"})

    assert "sk-secret" not in str(info.value)
    assert "ConnectionError" in str(info.value)


def test_pagination_has_a_hard_stop(http):
    url = "https://api.anthropic.com/v1/models"
    http.queue[url] = [_Resp({"data": [{"id": f"m{i}"}], "has_more": True, "last_id": f"m{i}"}) for i in range(50)]

    models = list_models("anthropic", {"ANTHROPIC_API_KEY": "k"})

    assert len(models) == model_listing._MAX_PAGES


def test_gemini_reports_its_400_for_an_invalid_key_as_a_rejected_key(http):
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    http.queue[url] = [_Resp({"error": {"status": "INVALID_ARGUMENT"}}, 400)]

    with pytest.raises(ModelListError, match="rejected this key"):
        list_models("gemini", {"GOOGLE_API_KEY": "bad"})


# --- Bedrock: a stubbed boto3 client (no AWS account is involved) -----------------------------------------


class _Bedrock:
    """Just the two calls the lister makes, answering from canned pages."""

    def __init__(self, profile_pages=(), models=(), profile_error=None, models_error=None):
        self.profile_pages, self.models = list(profile_pages), list(models)
        self.profile_error, self.models_error = profile_error, models_error
        self.profile_calls, self.model_calls = [], []

    def list_inference_profiles(self, **kwargs):
        self.profile_calls.append(kwargs)
        if self.profile_error:
            raise self.profile_error
        return self.profile_pages.pop(0)

    def list_foundation_models(self, **kwargs):
        self.model_calls.append(kwargs)
        if self.models_error:
            raise self.models_error
        return {"modelSummaries": self.models}


_BEDROCK_ENV = {"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "AKIA-x", "AWS_SECRET_ACCESS_KEY": "s3cret"}


def _client_error(code):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": "nope"}}, "Op")


def _use_bedrock(monkeypatch, stub):
    seen = {}

    def fake_client(service, **kwargs):
        seen.update(service=service, **kwargs)
        return stub

    monkeypatch.setattr("boto3.client", fake_client)
    return seen


def test_bedrock_lists_profiles_first_then_on_demand_models_without_duplicates(monkeypatch):
    stub = _Bedrock(
        profile_pages=[{"inferenceProfileSummaries": [
            {"inferenceProfileId": "us.anthropic.claude-x", "inferenceProfileName": "Claude X"},
        ]}],
        models=[
            {"modelId": "us.anthropic.claude-x", "modelName": "dup", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "amazon.titan", "modelName": "Titan", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "provisioned-only", "modelName": "P", "inferenceTypesSupported": ["PROVISIONED"]},
        ],
    )
    seen = _use_bedrock(monkeypatch, stub)

    models = list_models("bedrock", _BEDROCK_ENV)

    assert models == [{"id": "us.anthropic.claude-x", "label": "Claude X"}, {"id": "amazon.titan", "label": "Titan"}]
    assert stub.model_calls == [{"byOutputModality": "TEXT"}]
    assert seen["service"] == "bedrock" and seen["region_name"] == "us-east-1"


def test_bedrock_follows_inference_profile_pagination(monkeypatch):
    stub = _Bedrock(profile_pages=[
        {"inferenceProfileSummaries": [{"inferenceProfileId": "p1"}], "nextToken": "t2"},
        {"inferenceProfileSummaries": [{"inferenceProfileId": "p2"}]},
    ])
    _use_bedrock(monkeypatch, stub)

    assert [m["id"] for m in list_models("bedrock", _BEDROCK_ENV)] == ["p1", "p2"]
    assert stub.profile_calls == [{"maxResults": 1000}, {"maxResults": 1000, "nextToken": "t2"}]


def test_bedrock_falls_back_to_foundation_models_when_profiles_are_not_permitted(monkeypatch):
    stub = _Bedrock(
        profile_error=_client_error("AccessDeniedException"),
        models=[{"modelId": "amazon.titan", "modelName": "Titan", "inferenceTypesSupported": ["ON_DEMAND"]}],
    )
    _use_bedrock(monkeypatch, stub)

    assert [m["id"] for m in list_models("bedrock", _BEDROCK_ENV)] == ["amazon.titan"]


def test_bedrock_reports_an_aws_rejection_without_leaking_credentials(monkeypatch):
    _use_bedrock(monkeypatch, _Bedrock(profile_error=_client_error("UnrecognizedClientException"),
                                       models_error=_client_error("UnrecognizedClientException")))

    with pytest.raises(ModelListError) as info:
        list_models("bedrock", _BEDROCK_ENV)

    assert "UnrecognizedClientException" in str(info.value)
    assert "s3cret" not in str(info.value) and "AKIA-x" not in str(info.value)


@pytest.mark.parametrize("missing", ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
def test_bedrock_needs_all_three_credential_fields(missing):
    env = {k: v for k, v in _BEDROCK_ENV.items() if k != missing}

    with pytest.raises(ModelListError, match="saved for this provider"):
        list_models("bedrock", env)
