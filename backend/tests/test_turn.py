import threading

from pos.turn import run_turn


class _Llm:
    def __init__(self, pieces, usage=None, error=None):
        self.pieces, self.usage, self.error = pieces, usage or {}, error
        self.seen = []

    def stream(self, messages, cancel, usage=None):
        self.seen.append((messages, cancel))
        yield from self.pieces
        if self.error:
            raise self.error
        if usage is not None:
            usage.update(self.usage)


def test_collects_text_usage_and_timing():
    got = []
    stats = run_turn(_Llm(["a", "b"], {"total_tokens": 3}), [], threading.Event(), got.append)

    assert got == ["a", "b"] and stats.text == "ab"
    assert stats.usage == {"total_tokens": 3}
    assert stats.stopped is False
    assert stats.ttft is not None and stats.total >= stats.ttft


def test_on_piece_false_stops_early_and_keeps_what_arrived():
    stats = run_turn(_Llm(["a", "b", "c"]), [], threading.Event(), lambda p: p != "b")

    assert stats.stopped is True
    assert stats.text == "ab"


def test_none_from_on_piece_means_keep_going():
    stats = run_turn(_Llm(["a", "b"]), [], threading.Event(), lambda p: None)

    assert stats.text == "ab" and stats.stopped is False


def test_no_pieces_means_no_ttft_and_no_total():
    stats = run_turn(_Llm([]), [], threading.Event(), lambda p: None)

    assert stats.ttft is None and stats.total is None and stats.text == ""


def test_passes_messages_and_cancel_through_to_the_llm():
    llm = _Llm(["x"])
    cancel = threading.Event()
    messages = [{"role": "user", "content": "hi"}]

    run_turn(llm, messages, cancel, lambda p: None)

    assert llm.seen == [(messages, cancel)]


def test_an_llm_error_propagates_to_the_caller():
    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        run_turn(_Llm(["a"], error=RuntimeError("boom")), [], threading.Event(), lambda p: None)
