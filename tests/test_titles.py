"""titles.clean: keep a real title, drop one that only echoes the instructions."""

import pytest

from pos.titles import clean


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Read Notes.txt", "Read Notes.txt"),
        ('"Launch code revealed."', "Launch code revealed"),
        ("Title: Fox and the moon", "Fox and the moon"),
        ("**Lighthouse story**\nextra line", "Lighthouse story"),
        ("Title for this conversation: 3-8 words, sentence case. Return ONLY the title --", None),
        ("for the conversation. The conversation is: User asks to read notes.txt", None),
        ("", None),
        ("word " * 12, None),
    ],
)
def test_clean(raw: str, expected: str | None):
    assert clean(raw) == expected
