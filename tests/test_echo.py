import sys
import types

import numpy as np
import pytest

from asr_test.audio.echo import EchoControl


def test_headphones_mode_always_passes_mic_through():
    echo = EchoControl("headphones", mic_rate=16000, frame_size=512)
    assert echo.barge_in is True
    frame = np.ones(512, dtype=np.float32)
    assert np.array_equal(echo.process(frame, is_playing=True), frame)
    assert np.array_equal(echo.process(frame, is_playing=False), frame)


def test_duck_mode_drops_frames_only_while_playing():
    echo = EchoControl("duck", mic_rate=16000, frame_size=512)
    assert echo.barge_in is False
    frame = np.ones(512, dtype=np.float32)
    assert echo.process(frame, is_playing=True) is None
    assert np.array_equal(echo.process(frame, is_playing=False), frame)


def test_aec_mode_falls_back_to_duck_when_voiceclean_unavailable(monkeypatch):
    # voiceclean may or may not actually be installed in this environment
    # (it's an optional dependency) — force the import to fail either way,
    # to exercise the fallback path a user without the package hits.
    # `sys.modules[name] = None` is the standard way to make `import name`
    # raise ImportError regardless of what's actually installed.
    monkeypatch.setitem(sys.modules, "voiceclean", None)
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    assert echo.mode == "duck"
    assert echo.barge_in is False


class _FakeAecResult:
    def __init__(self, audio: bytes):
        self.audio = audio
        self.is_speech = True
        self.speech_prob = 0.9


class _FakeVoiceClean:
    """Mimics the real package's confirmed-live behavior: internal
    buffering on its own frame size, emitting variable-length (sometimes
    empty) chunks per call rather than echoing back exactly what was fed
    in — never 1:1 with the caller's frame size. Content-preserving
    (identity, just re-chunked) so tests can verify reassembly integrity,
    not just sizes."""

    _CHUNK_SAMPLES = 640  # observed real output chunk size, vs. our 512-sample input frames

    def __init__(self, sample_rate: int, **aec_kwargs):
        self.sample_rate = sample_rate
        self.aec_kwargs = aec_kwargs
        self.fed_references: list[bytes] = []
        self.processed: list[bytes] = []
        self._pending = b""
        self._calls = 0

    def feed_reference(self, pcm_bytes: bytes) -> None:
        self.fed_references.append(pcm_bytes)

    def process(self, pcm_bytes: bytes) -> _FakeAecResult:
        self.processed.append(pcm_bytes)
        self._calls += 1
        self._pending += pcm_bytes
        if self._calls == 1:
            return _FakeAecResult(audio=b"")  # first call: nothing ready yet, matches real behavior
        chunk_bytes = self._CHUNK_SAMPLES * 2
        if len(self._pending) >= chunk_bytes:
            out, self._pending = self._pending[:chunk_bytes], self._pending[chunk_bytes:]
            return _FakeAecResult(audio=out)
        return _FakeAecResult(audio=b"")


@pytest.fixture
def fake_voiceclean(monkeypatch):
    module = types.ModuleType("voiceclean")
    module.VoiceClean = _FakeVoiceClean
    monkeypatch.setitem(sys.modules, "voiceclean", module)
    return module


def test_aec_mode_calls_the_real_voiceclean_api_shape(fake_voiceclean):
    """Regression test for the bug this fixes: EchoControl used to call a
    voiceclean.AEC(...)/process(mic, ref) shape that doesn't exist in the
    real package (confirmed against voiceclean's own docs), so "aec" mode
    silently fell back to "duck" even with voiceclean installed. The real
    API is VoiceClean(sample_rate=...), feed_reference(pcm_bytes) for the
    bot's own audio, then process(mic_pcm_bytes) -> result.audio."""
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    assert echo.mode == "aec"
    assert echo.barge_in is True
    assert isinstance(echo.aec, _FakeVoiceClean)
    # Lowered from voiceclean's own 0.15 default per its docs' own guidance
    # for noisy/challenging echo conditions — see echo.py's comment.
    assert echo.aec.aec_kwargs.get("correlation_threshold") == 0.10

    ref_audio = np.full(512, 0.1, dtype=np.float32)
    echo.note_playback(ref_audio, source_rate=16000)
    assert len(echo.aec.fed_references) == 1

    mic_frame = np.full(512, 0.2, dtype=np.float32)
    echo.process(mic_frame, is_playing=True)

    assert len(echo.aec.processed) == 1


def test_aec_mode_skips_processing_when_not_playing(fake_voiceclean):
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    mic_frame = np.full(512, 0.2, dtype=np.float32)

    out = echo.process(mic_frame, is_playing=False)

    assert np.array_equal(out, mic_frame)
    assert echo.aec.processed == []


def test_aec_mode_reassembles_variable_size_voiceclean_output_into_fixed_frames(fake_voiceclean):
    """Regression test for a second bug found live: voiceclean doesn't
    return exactly frame_size samples per call (confirmed with the real
    package — feeding 512-sample frames produced 0 or 640 sample outputs,
    never 512). EchoControl.process() must buffer and re-chunk to exactly
    mic.size per call, matching WebSocketAudioSink's own drain pattern,
    or it hands VAD wrong-length or empty arrays."""
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    rng = np.random.default_rng(0)
    frames_in = [rng.uniform(-0.5, 0.5, 512).astype(np.float32) for _ in range(8)]

    outputs = []
    for frame in frames_in:
        echo.note_playback(frame, source_rate=16000)
        out = echo.process(frame, is_playing=True)
        if out is not None:
            assert out.shape == (512,)
            outputs.append(out)

    assert outputs  # buffering eventually produced at least one properly-sized frame

    original = np.concatenate(frames_in)
    reconstructed = np.concatenate(outputs)
    # The fake is content-preserving (identity, just re-chunked), so the
    # reassembled stream must match the original in order — some tail may
    # still be sitting in the internal buffer, hence the prefix compare.
    assert np.allclose(reconstructed, original[: reconstructed.size], atol=1e-3)


def test_aec_mode_clears_buffer_when_playback_stops(fake_voiceclean):
    echo = EchoControl("aec", mic_rate=16000, frame_size=512)
    frame = np.full(512, 0.2, dtype=np.float32)

    for _ in range(4):  # get some cleaned audio sitting in the internal buffer
        echo.note_playback(frame, source_rate=16000)
        echo.process(frame, is_playing=True)
    assert echo._aec_out_buf.size > 0

    echo.process(frame, is_playing=False)

    assert echo._aec_out_buf.size == 0
