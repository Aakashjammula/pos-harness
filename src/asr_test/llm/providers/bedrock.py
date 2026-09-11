"""AWS Bedrock. See
https://docs.langchain.com/oss/python/langchain/models#aws-bedrock.
No single API key -- needs AWS-style credentials (access key id,
secret access key, region) instead, see ProviderConfig's aws_* fields."""

from __future__ import annotations

import os

from langchain_aws import ChatBedrock

from .base import LlmProviderBase, ProviderConfig
from .registry import register

_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
_DEFAULT_REGION = "us-east-1"


@register
class BedrockProvider(LlmProviderBase):
    name = "bedrock"
    priority = 40

    def detect(self, env) -> bool:
        return bool(env.get("AWS_ACCESS_KEY_ID")) and bool(env.get("AWS_SECRET_ACCESS_KEY"))

    def resolve(self, model_override: str | None, env) -> ProviderConfig:
        access_key = env["AWS_ACCESS_KEY_ID"]
        secret_key = env["AWS_SECRET_ACCESS_KEY"]
        region = env.get("AWS_REGION", _DEFAULT_REGION)
        model = model_override or env.get("BEDROCK_MODEL", _DEFAULT_MODEL)
        return ProviderConfig(
            name=self.name,
            model=model,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region=region,
        )

    def build_model(self, provider: ProviderConfig, **model_kwargs):
        # ChatBedrock doesn't accept stream_usage either -- same
        # not-a-real-param situation as Gemini.
        model_kwargs.pop("stream_usage", None)
        return ChatBedrock(
            model=provider.model,
            region_name=provider.aws_region,
            aws_access_key_id=provider.aws_access_key_id,
            aws_secret_access_key=provider.aws_secret_access_key,
            **model_kwargs,
        )

    def price_for(self, provider: ProviderConfig) -> tuple[float, float] | None:
        env_in = os.environ.get("BEDROCK_PRICE_INPUT_PER_1K")
        env_out = os.environ.get("BEDROCK_PRICE_OUTPUT_PER_1K")
        if env_in is not None and env_out is not None:
            return (float(env_in), float(env_out))
        return None

    def context_window_for(self, provider: ProviderConfig) -> int | None:
        env_val = os.environ.get("BEDROCK_CONTEXT_WINDOW")
        if env_val is not None:
            return int(env_val)
        return None
