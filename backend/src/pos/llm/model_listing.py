"""Ask a provider which models a given credential can actually use.

Nothing about model names is assumed: the user saves a key (and a URL where
one is needed), and this returns what that account really has, so they pick
from a real list instead of relying on a hardcoded default that may not exist.

Each provider has its own listing API (verified against the vendor docs):

  local       GET {base_url}/models                        OpenAI-compatible
  openai      GET https://api.openai.com/v1/models         also lists embeddings/audio/image
  anthropic   GET https://api.anthropic.com/v1/models      paginated with after_id
  gemini      GET .../v1beta/models                         keep generateContent models only
  openrouter  GET https://openrouter.ai/api/v1/models      keep text-output + tool-capable
  bedrock     boto3 list_inference_profiles (paginated) + list_foundation_models.
              Both calls and their parameters are confirmed against botocore's own
              service model; the behaviour is covered with a stubbed client, not a
              live AWS account.
  azure       not listable with an API key: deployment names are user-defined

`env` uses the same variable names as the credential store (PROVIDER_FIELDS
in pos.auth.routes), so a stored credential can be passed straight in. Keys
go in headers, never in URLs, so they cannot surface in logged URLs or
exception text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import requests

from pos.net_policy import UnsafeUrl, check_user_url

_TIMEOUT = 10
_MAX_PAGES = 20   # a hard stop on pagination loops


class ModelListError(Exception):
    """A listing failed in a way worth showing the user. The message never
    contains a credential."""


# {"id", "label", "chat"} plus "context_window" (input tokens) when the provider says. "chat": answers in
# text, so worth offering by default. Where the API has no context field (OpenAI's does not) the key is left
# out rather than guessed from a table that would go stale.
Model = dict


def _with_window(model: Model, *candidates) -> Model:
    """Add context_window from the first candidate that is a positive integer."""
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return {**model, "context_window": value}
    return model


def _get_json(
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, object] | None = None,
    bad_key_statuses: tuple[int, ...] = (401, 403),
) -> dict:
    try:
        resp = requests.get(url, headers=dict(headers), params=params, timeout=_TIMEOUT, allow_redirects=False)
    except requests.RequestException as e:
        # str(e) can embed the URL; keep only the exception type.
        raise ModelListError(f"couldn't reach the provider ({type(e).__name__})") from None
    if resp.status_code in bad_key_statuses:
        raise ModelListError("the provider rejected this key (unauthorized)")
    if 300 <= resp.status_code < 400:   # never followed (see net_policy): say so rather than reading it as success
        raise ModelListError("the server redirected the request, and redirects are not followed")
    if resp.status_code >= 400:
        raise ModelListError(f"the provider returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        raise ModelListError("the provider returned something that isn't JSON") from None


def _need(env: Mapping[str, str], name: str, what: str) -> str:
    value = env.get(name)
    if not value:
        raise ModelListError(f"no {what} saved for this provider")
    return value


# --- OpenAI-compatible -------------------------------------------------------

# A provider's model list mixes in models that cannot hold a conversation: speech synthesis and
# transcription, image and video generation, embeddings, rerankers, moderation, realtime/live
# audio. This app talks to a model in text (multimodal INPUT such as images is fine), so only
# models that answer in text are offered. The providers that describe capabilities in their API
# (LM Studio, OpenRouter) are judged by that; the rest by name, which is what this is for.
# Longer stems match anywhere; short ones only as whole words ("tts" must not match "settings").
_NOT_CHAT_STEMS = (
    "embed", "whisper", "transcri", "speech", "image", "dall-e", "moderation", "rerank", "realtime",
    "audio", "davinci", "babbage",
)
_NOT_CHAT_WORDS = ("tts", "live", "veo", "sora", "aqa")
_NOT_CHAT = re.compile(
    "(" + "|".join(_NOT_CHAT_STEMS) + r")|(?:^|[-_/.:])(?:" + "|".join(_NOT_CHAT_WORDS) + r")(?:$|[-_/.:])",
    re.IGNORECASE,
)


def _is_chat_model(model_id: str) -> bool:
    return not _NOT_CHAT.search(model_id)


# Gemini's models.list has NO field for what a model outputs and no filter (checked against the API
# reference), and it returns image, music, robotics, agent and speech models beside the chat ones -- all
# with generateContent. So this is a heuristic, and the list is curated by family: Gemini and Gemma,
# minus variants whose name marks a special purpose. Whatever it hides is still one click away
# ("show all models"), so a wrong guess is an inconvenience, never a block.
_GEMINI_FAMILY = re.compile(r"^(gemini|gemma)-")
_GEMINI_SPECIAL_PURPOSE = {
    "image", "imagen", "tts", "live", "audio", "native", "embed", "embedding", "veo",
    "robotics", "computer", "customtools", "omni",
}


def _is_gemini_chat(model_id: str) -> bool:
    if not _GEMINI_FAMILY.match(model_id):
        return False
    return not (set(re.split(r"[-._]", model_id.lower())) & _GEMINI_SPECIAL_PURPOSE)


def _list_local(env: Mapping[str, str]) -> list[Model]:
    base = _need(env, "LOCAL_BASE_URL", "server URL").rstrip("/")
    try:
        check_user_url("LOCAL_BASE_URL", base)
    except UnsafeUrl as e:
        raise ModelListError(f"that server URL is not allowed: {e}") from None
    key = env.get("LOCAL_API_KEY") or "lm-studio"
    headers = {"Authorization": f"Bearer {key}"}
    # LM Studio labels each model llm / vlm / embeddings: exact, so use it when it is there.
    try:
        typed = _get_json(f"{base.removesuffix('/v1')}/api/v0/models", headers=headers).get("data", [])
    except ModelListError:
        typed = []                                          # not LM Studio: fall through to the name filter
    if any("type" in m for m in typed):
        return [   # what is loaded right now beats the model's theoretical maximum
            _with_window(
                {"id": m["id"], "label": m["id"], "chat": m.get("type") in ("llm", "vlm")},
                m.get("loaded_context_length"), m.get("max_context_length"),
            )
            for m in typed
            if m.get("id")
        ]
    data = _get_json(f"{base}/models", headers=headers)
    return [
        {"id": m["id"], "label": m["id"], "chat": _is_chat_model(m["id"])} for m in data.get("data", []) if m.get("id")
    ]


def _list_openai(env: Mapping[str, str]) -> list[Model]:
    key = _need(env, "OPENAI_API_KEY", "API key")
    data = _get_json("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})
    rows = [m for m in data.get("data", []) if m.get("id")]
    rows.sort(key=lambda m: m.get("created", 0), reverse=True)   # newest first
    return [{"id": m["id"], "label": m["id"], "chat": _is_chat_model(m["id"])} for m in rows]


# --- Anthropic ---------------------------------------------------------------


def _list_anthropic(env: Mapping[str, str]) -> list[Model]:
    key = _need(env, "ANTHROPIC_API_KEY", "API key")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    models: list[Model] = []
    after: str | None = None
    for _ in range(_MAX_PAGES):
        params: dict[str, object] = {"limit": 1000}
        if after:
            params["after_id"] = after
        data = _get_json("https://api.anthropic.com/v1/models", headers=headers, params=params)
        models += [
            _with_window({"id": m["id"], "label": m.get("display_name") or m["id"], "chat": True},
                         m.get("max_input_tokens"))
            for m in data.get("data", [])
            if m.get("id")
        ]
        if not data.get("has_more") or not data.get("last_id"):
            break
        after = data["last_id"]
    return models   # the API already lists the newest first


# --- Google Gemini -----------------------------------------------------------


def _list_gemini(env: Mapping[str, str]) -> list[Model]:
    key = _need(env, "GOOGLE_API_KEY", "API key")
    models: list[Model] = []
    token: str | None = None
    for _ in range(_MAX_PAGES):
        params: dict[str, object] = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        data = _get_json(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key},
            params=params,
            # Gemini answers an invalid key with 400 (API_KEY_INVALID), not 401. Our
            # request is otherwise fixed, so a 400 here means the key.
            bad_key_statuses=(400, 401, 403),
        )
        for m in data.get("models", []):
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue   # embeddings, AQA, etc.
            model_id = m.get("name", "").removeprefix("models/")
            if model_id:   # speech, image, music and agent models support generateContent too: see _is_gemini_chat
                models.append(_with_window(
                    {"id": model_id, "label": m.get("displayName") or model_id, "chat": _is_gemini_chat(model_id)},
                    m.get("inputTokenLimit"),
                ))
        token = data.get("nextPageToken")
        if not token:
            break
    return models


# --- OpenRouter --------------------------------------------------------------


def _list_openrouter(env: Mapping[str, str]) -> list[Model]:
    key = _need(env, "OPENROUTER_API_KEY", "API key")
    data = _get_json("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
    models: list[Model] = []
    for m in data.get("data", []):
        if not m.get("id"):
            continue
        # OpenRouter describes what a model outputs and which parameters it supports, so use that:
        # chat = answers in text only (image input is fine) and, because this app binds tools,
        # can call them (only judged when the field is present).
        outputs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
        params = m.get("supported_parameters")
        chat = set(outputs) == {"text"} and (params is None or "tools" in params)
        models.append(_with_window({"id": m["id"], "label": m.get("name") or m["id"], "chat": chat},
                                   m.get("context_length")))
    return models


# --- AWS Bedrock -------------------------------------------------------------


def _list_bedrock(env: Mapping[str, str]) -> list[Model]:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    region = _need(env, "AWS_REGION", "region")
    client = boto3.client(
        "bedrock",
        region_name=region,
        aws_access_key_id=_need(env, "AWS_ACCESS_KEY_ID", "access key id"),
        aws_secret_access_key=_need(env, "AWS_SECRET_ACCESS_KEY", "secret access key"),
    )
    models: list[Model] = []
    try:
        # Cross-region inference profiles (ids like "us.anthropic.claude-...") are
        # what recent models require and are not returned by list_foundation_models.
        try:
            token: str | None = None
            for _ in range(_MAX_PAGES):
                page = client.list_inference_profiles(maxResults=1000, **({"nextToken": token} if token else {}))
                for p in page.get("inferenceProfileSummaries", []):
                    profile_id = p["inferenceProfileId"]
                    models.append(
                        {"id": profile_id, "label": p.get("inferenceProfileName") or profile_id, "chat": True}
                    )
                token = page.get("nextToken")
                if not token:
                    break
        except (AttributeError, ClientError):
            pass   # older boto3 / no permission: fall back to foundation models only
        seen = {m["id"] for m in models}
        for m in client.list_foundation_models(byOutputModality="TEXT").get("modelSummaries", []):
            if m["modelId"] not in seen and "ON_DEMAND" in m.get("inferenceTypesSupported", []):
                models.append({"id": m["modelId"], "label": m.get("modelName") or m["modelId"], "chat": True})
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "error")
        raise ModelListError(f"AWS rejected the request ({code})") from None
    except BotoCoreError as e:
        raise ModelListError(f"couldn't reach AWS ({type(e).__name__})") from None
    return models


def _list_azure(env: Mapping[str, str]) -> list[Model]:
    raise ModelListError(
        "Azure deployments can't be listed with an API key -- deployment names are yours to choose; "
        "enter the deployment name instead"
    )


_LISTERS: dict[str, Callable[[Mapping[str, str]], list[Model]]] = {
    "local": _list_local,
    "openai": _list_openai,
    "anthropic": _list_anthropic,
    "gemini": _list_gemini,
    "openrouter": _list_openrouter,
    "bedrock": _list_bedrock,
    "azure": _list_azure,
}


def supported_providers() -> list[str]:
    return sorted(_LISTERS)


def list_models(provider: str, env: Mapping[str, str], include_all: bool = False) -> list[Model]:
    """The models `env`'s credentials can use with `provider`. By default only those that answer in
    text ({"id", "label"}); with include_all every model comes back with a "chat" flag, so the UI can
    offer the rest on request. Blocking (uses `requests`/boto3) -- call it off the event loop."""
    lister = _LISTERS.get(provider)
    if lister is None:
        raise ModelListError(f"{provider!r} has no models to list")
    models = lister(env)
    if include_all:
        return models
    return [{"id": m["id"], "label": m["label"]} for m in models if m["chat"]]
