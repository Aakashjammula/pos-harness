from asr_test.llm.pricing import estimate_cost, price_for


def test_local_is_always_free(monkeypatch):
    monkeypatch.delenv("LOCAL_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("LOCAL_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("local", "anything") == (0.0, 0.0)
    assert estimate_cost("local", "anything", input_tokens=999999, output_tokens=999999) == 0.0


def test_known_openai_model_uses_built_in_table(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    prices = price_for("openai", "gpt-4o-mini")

    assert prices == (0.15, 0.60)


def test_unknown_model_with_no_override_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("openai", "some-unlisted-model") is None
    assert estimate_cost("openai", "some-unlisted-model", 1000, 1000) is None


def test_azure_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("AZURE_PRICE_INPUT_PER_1K", raising=False)
    monkeypatch.delenv("AZURE_PRICE_OUTPUT_PER_1K", raising=False)

    assert price_for("azure", "my-deployment") is None


def test_env_override_wins_over_built_in_table(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    assert price_for("openai", "gpt-4o-mini") == (1.0, 2.0)


def test_env_override_enables_pricing_for_azure(monkeypatch):
    monkeypatch.setenv("AZURE_PRICE_INPUT_PER_1K", "3.0")
    monkeypatch.setenv("AZURE_PRICE_OUTPUT_PER_1K", "4.0")

    assert price_for("azure", "my-deployment") == (3.0, 4.0)


def test_estimate_cost_computes_weighted_sum(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICE_INPUT_PER_1K", "1.0")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1K", "2.0")

    # 2000 input tokens @ $1.0/1K = $2.0; 500 output tokens @ $2.0/1K = $1.0
    cost = estimate_cost("openai", "gpt-4o-mini", input_tokens=2000, output_tokens=500)

    assert cost == 3.0
