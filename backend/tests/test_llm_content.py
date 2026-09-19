from pos.llm.content import content_text


def test_string_and_none():
    assert content_text("hi") == "hi"
    assert content_text(None) == ""


def test_text_blocks_are_joined_and_non_text_blocks_ignored():
    blocks = [
        {"type": "text", "text": "Hel"},
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "lo"},
        "!",
    ]
    assert content_text(blocks) == "Hello!"


def test_a_text_block_with_no_text_is_empty_not_an_error():
    assert content_text([{"type": "text"}, {"type": "text", "text": None}]) == ""
