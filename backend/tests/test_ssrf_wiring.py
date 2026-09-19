"""The SSRF policy must bite where URLs are actually used, not just exist."""

import base64
import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from pos import config
from pos.auth.routes import build_auth_router
from pos.auth.store import UserStore
from pos.db import create_pool, init_schema
from pos.llm.model_listing import ModelListError, list_models
from pos.llm.providers.azure import AzureProvider
from pos.llm.providers.local import LocalProvider

_pool = None
METADATA = "http://169.254.169.254/latest/meta-data"


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def _client():
    global _pool
    if _pool is None:
        _pool = create_pool(os.environ["TEST_DATABASE_URL"], max_size=5)
        init_schema(_pool)
    with _pool.connection() as conn:
        conn.execute("TRUNCATE users, magic_link_tokens RESTART IDENTITY CASCADE")
    app = FastAPI()
    app.include_router(build_auth_router(UserStore(_pool)))
    client = TestClient(app)
    client.post(
        "/auth/signup",
        json={"name": "A", "username": "user_a", "email": "a@test.com", "password": "pw-12345678"},
    )
    return client


# --- saving ---------------------------------------------------------------------------------------------

def test_saving_the_metadata_address_as_a_server_url_is_refused_even_when_private_is_allowed(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", True)
    client = _client()

    resp = client.put("/credentials/local", json={"local_base_url": METADATA})

    assert resp.status_code == 422 and "LOCAL_BASE_URL" in resp.json()["detail"]
    assert client.get("/credentials").json()["configured"] == []        # nothing was stored


def test_a_private_server_url_is_accepted_only_when_the_operator_allows_it(monkeypatch):
    client = _client()
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)
    assert client.put("/credentials/local", json={"local_base_url": "http://192.168.1.5:1234/v1"}).status_code == 422

    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", True)
    assert client.put("/credentials/local", json={"local_base_url": "http://192.168.1.5:1234/v1"}).status_code == 200


def test_a_public_server_url_is_always_accepted(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)
    assert _client().put("/credentials/local", json={"local_base_url": "https://llm.example.com/v1"}).status_code == 200


def test_an_azure_endpoint_must_be_public_https_regardless_of_the_private_setting(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", True)
    client = _client()
    for bad in ("http://myres.openai.azure.com/", "https://192.168.1.5/", "https://127.0.0.1/"):
        assert client.put("/credentials/azure", json={"azure_endpoint": bad}).status_code == 422, bad
    good = {"azure_endpoint": "https://myres.openai.azure.com/"}
    assert client.put("/credentials/azure", json=good).status_code == 200


def test_the_error_does_not_reveal_what_the_hostname_resolved_to(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)
    monkeypatch.setattr("pos.net_policy._resolve", lambda host, port: ["10.9.8.7"])

    resp = _client().put("/credentials/local", json={"local_base_url": "http://internal-thing.corp/v1"})

    assert resp.status_code == 422 and "10.9.8.7" not in resp.text and "internal-thing" not in resp.text


# --- using ----------------------------------------------------------------------------------------------

def test_model_listing_refuses_before_making_any_request(monkeypatch):
    calls = []
    monkeypatch.setattr("pos.llm.model_listing.requests.get", lambda *a, **k: calls.append(a) or None)

    with pytest.raises(ModelListError, match="not allowed"):
        list_models("local", {"LOCAL_BASE_URL": METADATA})

    assert calls == []                              # not even a probe went out


def test_model_listing_never_follows_redirects(monkeypatch):
    seen = {}

    class Resp:
        status_code = 302

        def json(self):
            return {}

    def fake_get(url, **kw):
        seen.update(kw)
        return Resp()

    monkeypatch.setattr("pos.llm.model_listing.requests.get", fake_get)

    with pytest.raises(ModelListError, match="redirect"):
        list_models("local", {"LOCAL_BASE_URL": "http://llm.example.com/v1"})
    assert seen["allow_redirects"] is False


def test_the_local_provider_refuses_an_unsafe_url_when_building_a_session(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)

    with pytest.raises(RuntimeError, match="not allowed"):
        LocalProvider().resolve("m", {"LOCAL_BASE_URL": "http://127.0.0.1:1234/v1"})
    with pytest.raises(RuntimeError, match="not allowed"):
        LocalProvider().resolve("m", {"LOCAL_BASE_URL": METADATA})


def test_an_operator_configured_private_url_keeps_working_without_the_opt_in(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", False)
    monkeypatch.setenv("LOCAL_BASE_URL", "http://host.docker.internal:1234/v1")
    monkeypatch.setattr("pos.net_policy._resolve", lambda host, port: ["172.17.0.1"])

    provider = LocalProvider().resolve("m", {"LOCAL_BASE_URL": "http://host.docker.internal:1234/v1"})

    assert provider.base_url == "http://host.docker.internal:1234/v1"


def test_the_local_chat_client_does_not_follow_redirects(monkeypatch):
    captured = {}
    monkeypatch.setattr("pos.llm.providers.local.ChatOpenAI", lambda **kw: captured.update(kw) or "model")
    provider = LocalProvider().resolve("m", {"LOCAL_BASE_URL": "http://llm.example.com/v1"})

    LocalProvider().build_model(provider, max_tokens=5)

    assert captured["http_client"].follow_redirects is False


def test_the_azure_provider_refuses_a_non_azure_endpoint():
    env = {
        "AZURE_OPENAI_API_KEY": "k",
        "AZURE_OPENAI_ENDPOINT": "http://169.254.169.254/",
        "AZURE_OPENAI_DEPLOYMENT": "d",
    }
    with pytest.raises(RuntimeError, match="not allowed"):
        AzureProvider().resolve(None, env)
