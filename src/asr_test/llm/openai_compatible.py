from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from openai import OpenAI

from ..interfaces.llm import LlmBase


class OpenAiCompatibleLlm(LlmBase):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",   # LM Studio OpenAI-compatible server
        api_key: str = "lm-studio",                    # unused by LM Studio, required by the SDK
        model: str = "lfm2.5-230m",
        system_prompt: str = (
            "You are a concise voice assistant. Answer in one or two short sentences. "
            "Plain text only — no markdown, lists, or emoji. Your words are spoken aloud."
        ),
        max_tokens: int = 120,
        timeout: float = 30,
        warmup: bool = True,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.last_ttft: float | None = None
        self.last_total: float | None = None

        if warmup:
            t0 = time.perf_counter()
            try:
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    timeout=self.timeout,
                )
                print(f"  llm warm-up: {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"  llm warm-up failed ({e}) — is LM Studio running?")

    def stream(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        t_start = time.perf_counter()
        self.last_ttft = None
        self.last_total = None

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=0.7,
            stream=True,
            timeout=self.timeout,
        )
        for chunk in completion:
            if cancel.is_set():
                break
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                if self.last_ttft is None:
                    self.last_ttft = time.perf_counter() - t_start
                yield piece

        self.last_total = time.perf_counter() - t_start
