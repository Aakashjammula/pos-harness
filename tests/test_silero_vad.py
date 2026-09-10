from unittest.mock import MagicMock

from asr_test.vad.silero import SileroVad


def test_threshold_and_tuning_params_pass_through_to_vad_iterator(monkeypatch):
    monkeypatch.setattr("asr_test.vad.silero.load_silero_vad", lambda onnx: "fake-model")
    captured = {}

    class _SpyVADIterator:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

        def __call__(self, frame, return_seconds=False):
            return None

    monkeypatch.setattr("asr_test.vad.silero.VADIterator", _SpyVADIterator)

    vad = SileroVad(sample_rate=16000, min_silence_ms=700, speech_pad_ms=200, threshold=0.35)

    assert captured["model"] == "fake-model"
    assert captured["threshold"] == 0.35
    assert captured["sampling_rate"] == 16000
    assert captured["min_silence_duration_ms"] == 700
    assert captured["speech_pad_ms"] == 200

    # __call__ still delegates to the (fake) iterator correctly
    import numpy as np

    assert vad(np.zeros(512, dtype=np.float32)) is None


def test_default_threshold_matches_silero_vad_iterators_own_default(monkeypatch):
    monkeypatch.setattr("asr_test.vad.silero.load_silero_vad", lambda onnx: "fake-model")
    captured = {}

    class _SpyVADIterator:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

        def __call__(self, frame, return_seconds=False):
            return None

    monkeypatch.setattr("asr_test.vad.silero.VADIterator", _SpyVADIterator)

    SileroVad()

    assert captured["threshold"] == 0.5
