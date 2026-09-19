import json

from pos.tts.kokoro import _DEFAULT_REPO, _PINNED_REVISION, KokoroTts, _revision_for


def test_the_default_repo_is_pinned_to_the_tested_revision():
    assert _revision_for(_DEFAULT_REPO, None) == _PINNED_REVISION
    assert len(_PINNED_REVISION) == 40 and set(_PINNED_REVISION) <= set("0123456789abcdef")


def test_an_explicit_revision_wins_and_a_custom_repo_is_left_alone():
    assert _revision_for(_DEFAULT_REPO, "abc123") == "abc123"
    assert _revision_for("someone/else", None) is None


def test_listing_voices_downloads_the_pinned_revision(monkeypatch, tmp_path):
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps({"af_bella": [], "am_adam": []}))
    seen = {}

    def fake_download(repo, filename, **kwargs):
        seen.update(repo=repo, filename=filename, **kwargs)
        return str(voices)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert KokoroTts.list_voices() == ["af_bella", "am_adam"]
    assert seen["revision"] == _PINNED_REVISION
