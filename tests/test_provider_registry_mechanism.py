import pytest

from asr_test.llm.providers import registry
from asr_test.llm.providers.base import LlmProviderBase, ProviderConfig
from asr_test.llm.providers.registry import (
    build_model,
    context_window_for,
    estimate_cost,
    price_for,
    register,
    resolve_provider,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    # The real registry (Task 5 onward) is populated once at import time
    # and shared process-wide -- these tests register throwaway fakes
    # and must not leak them into (or see) that real registry. Patch the
    # module attribute itself (not a name-imported snapshot of the dict)
    # so register()/resolve_provider()/etc. -- which all look it up by
    # name in registry.py's own module namespace -- see the reset.
    monkeypatch.setattr(registry, "_REGISTRY", {})


class _FakeProvider(LlmProviderBase):
    name = "fake"
    priority = 100
    detected = True

    def detect(self):
        return self.detected

    def resolve(self, model_override):
        return ProviderConfig(name=self.name, model=model_override or "fake-default-model")

    def build_model(self, provider, **model_kwargs):
        self.build_calls.append((provider, model_kwargs))
        return "fake-model-instance"

    def price_for(self, provider):
        return (1.0, 2.0)

    def context_window_for(self, provider):
        return 4096


def test_register_and_resolve_dispatches_to_the_only_registered_provider():
    register(_FakeProvider)

    provider = resolve_provider()

    assert provider.name == "fake"
    assert provider.model == "fake-default-model"


def test_resolve_provider_checks_providers_in_priority_order():
    class _HighPriority(_FakeProvider):
        name = "high"
        priority = 0

        def resolve(self, model_override):
            return ProviderConfig(name="high", model="high-model")

    class _LowPriority(_FakeProvider):
        name = "low"
        priority = 50

        def resolve(self, model_override):
            return ProviderConfig(name="low", model="low-model")

    register(_LowPriority)
    register(_HighPriority)

    provider = resolve_provider()

    assert provider.name == "high"  # lower priority number checked first


def test_resolve_provider_skips_providers_that_dont_detect():
    class _NotDetected(_FakeProvider):
        name = "not-detected"
        priority = 0
        detected = False

    class _Fallback(_FakeProvider):
        name = "fallback"
        priority = 100

    register(_NotDetected)
    register(_Fallback)

    provider = resolve_provider()

    assert provider.name == "fallback"


def test_build_model_dispatches_to_the_matching_registered_provider():
    _FakeProvider.build_calls = []
    register(_FakeProvider)
    fake_instance = registry._REGISTRY["fake"]
    provider_config = ProviderConfig(name="fake", model="m")

    result = build_model(provider_config, some_kwarg=1)

    assert result == "fake-model-instance"
    assert fake_instance.build_calls == [(provider_config, {"some_kwarg": 1})]


def test_price_for_and_context_window_for_dispatch_to_matching_provider():
    register(_FakeProvider)
    provider_config = ProviderConfig(name="fake", model="m")

    assert price_for(provider_config) == (1.0, 2.0)
    assert context_window_for(provider_config) == 4096


def test_estimate_cost_computes_weighted_sum_from_price_for():
    register(_FakeProvider)
    provider_config = ProviderConfig(name="fake", model="m")

    # price_for returns (1.0, 2.0): 2000 input @ $1/1K = $2, 500 output @ $2/1K = $1
    cost = estimate_cost(provider_config, input_tokens=2000, output_tokens=500)

    assert cost == 3.0


def test_estimate_cost_returns_none_when_unpriced():
    class _Unpriced(_FakeProvider):
        name = "unpriced"
        priority = 100

        def price_for(self, provider):
            return None

    register(_Unpriced)
    provider_config = ProviderConfig(name="unpriced", model="m")

    assert estimate_cost(provider_config, input_tokens=1000, output_tokens=1000) is None
