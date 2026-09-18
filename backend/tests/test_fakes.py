import threading

from fakes import FakeLlm


def test_fake_llm_ignores_usage_by_default():
    llm = FakeLlm("hi there")
    usage = {}

    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage == {}


def test_fake_llm_reports_configured_fake_usage():
    fake_usage = {"provider": "local", "model": "x", "input_tokens": 5, "output_tokens": 3}
    llm = FakeLlm("hi there", fake_usage=fake_usage)
    usage = {}

    list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), usage))

    assert usage == fake_usage


def test_fake_llm_works_with_usage_none():
    llm = FakeLlm("hi there")

    result = list(llm.stream([{"role": "user", "content": "hi"}], threading.Event(), None))

    assert result == ["hi ", "there "]
