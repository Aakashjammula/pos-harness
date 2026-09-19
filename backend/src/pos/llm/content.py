"""Provider replies carry `content` as a str or -- Gemini, and Anthropic with
tools or thinking -- as a list of content blocks. Everything downstream (the
sentence chunker feeding TTS, the SSE token stream, titles) wants plain text,
and concatenating a list onto a str raises TypeError."""

from __future__ import annotations


def content_text(content) -> str:
    """The text of a message's content: a str as-is, the text blocks of a block
    list joined, anything else (thinking, tool-use blocks, None) ignored."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)
