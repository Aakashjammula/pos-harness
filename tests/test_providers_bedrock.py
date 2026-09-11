import os

from asr_test.llm.providers.base import ProviderConfig
from asr_test.llm.providers.bedrock import BedrockProvider


def _bedrock_provider(model="us.anthropic.claude-sonnet-4-6"):
    return ProviderConfig(
        name="bedrock", model=model,
        aws_access_key_id="AKIAFAKE", aws_secret_access_key="fakefakefake", aws_region="us-east-1",
    )


def test_detect_true_when_both_credentials_set(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefake")

    assert BedrockProvider().detect(os.environ) is True


def test_detect_false_when_access_key_missing(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefake")

    assert BedrockProvider().detect(os.environ) is False


def test_detect_false_when_secret_key_missing(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    assert BedrockProvider().detect(os.environ) is False


def test_resolve_uses_default_model_and_region_when_not_set(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefake")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)

    provider = BedrockProvider().resolve(model_override=None, env=os.environ)

    assert provider.name == "bedrock"
    assert provider.aws_access_key_id == "AKIAFAKE"
    assert provider.aws_secret_access_key == "fakefakefake"
    assert provider.aws_region == "us-east-1"
    assert provider.model == "us.anthropic.claude-sonnet-4-6"


def test_resolve_region_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefake")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    provider = BedrockProvider().resolve(model_override=None, env=os.environ)

    assert provider.aws_region == "eu-west-1"


def test_resolve_model_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fakefakefake")
    monkeypatch.setenv("BEDROCK_MODEL", "us.anthropic.claude-opus-4-6")

    provider = BedrockProvider().resolve(model_override="us.anthropic.claude-haiku-4-6", env=os.environ)

    assert provider.model == "us.anthropic.claude-haiku-4-6"


def test_price_for_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("BEDROCK_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("BEDROCK_PRICE_OUTPUT_PER_1K", raising=False)

    assert BedrockProvider().price_for(_bedrock_provider()) is None


def test_price_for_env_override_enables_pricing(monkeypatch):
    monkeypatch.setenv("BEDROCK_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("BEDROCK_PRICE_OUTPUT_PER_1K", "15.0")

    assert BedrockProvider().price_for(_bedrock_provider()) == (3.0, 15.0)


def test_context_window_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("BEDROCK_CONTEXT_WINDOW", raising=False)

    assert BedrockProvider().context_window_for(_bedrock_provider()) is None


def test_context_window_env_override_enables_it(monkeypatch):
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "200000")

    assert BedrockProvider().context_window_for(_bedrock_provider()) == 200000


def test_build_model_passes_expected_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_chat_bedrock(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.bedrock.ChatBedrock", fake_chat_bedrock)

    result = BedrockProvider().build_model(
        _bedrock_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True
    )

    assert result == "the-model"
    assert captured_kwargs["model"] == "us.anthropic.claude-sonnet-4-6"
    assert captured_kwargs["region_name"] == "us-east-1"
    assert captured_kwargs["aws_access_key_id"] == "AKIAFAKE"
    assert captured_kwargs["aws_secret_access_key"] == "fakefakefake"


def test_build_model_strips_unsupported_stream_usage_kwarg(monkeypatch):
    captured_kwargs = {}

    def fake_chat_bedrock(**kwargs):
        captured_kwargs.update(kwargs)
        return "the-model"

    monkeypatch.setattr("asr_test.llm.providers.bedrock.ChatBedrock", fake_chat_bedrock)

    BedrockProvider().build_model(_bedrock_provider(), max_tokens=120, temperature=0.7, timeout=30, stream_usage=True)

    assert "stream_usage" not in captured_kwargs
