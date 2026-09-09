from __future__ import annotations

import json
import time

import numpy as np

from ..interfaces.tts import TtsBase


class KokoroTts(TtsBase):
    """
    Two things dominate Kokoro's realtime behaviour:
      1. warm-up — the first InferenceSession.run() pays graph optimization
         and memory-arena allocation, which can be seconds
      2. SessionOptions — thread count and optimization level are untuned
         by default

    Precision: fp32 ("kokoro-base-onnx"), not int8. Benchmarked on an
    i5-1235U (12th-gen Intel mobile — AVX-512/VNNI disabled at the
    microcode level on this whole generation, so int8 kernels get no
    hardware acceleration): int8 held ~1.5x realtime regardless of thread
    count (4/8/10 threads all ~1.5-1.7 RTF), while fp32 at 8 threads ran at
    0.49 RTF — 3x faster. Without VNNI, "int8" is a pessimization, not an
    optimization. Re-benchmark before changing this on different hardware.
    """

    def __init__(
        self,
        repo: str = "NeuML/kokoro-base-onnx",
        voice: str = "af_bella",
        speed: float = 1.0,
        sample_rate: int = 24000,
        threads: int = 8,               # i5-1235U: physical core count; see class docstring
        warmup: bool = True,
    ):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from ttstokenizer import IPATokenizer

        self.sample_rate = sample_rate
        self.speed = speed

        model_path = hf_hub_download(repo, "model.onnx")
        voices_path = hf_hub_download(repo, "voices.json")

        with open(voices_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
        if voice not in voices:
            raise ValueError(f"voice '{voice}' missing; have {sorted(voices)[:12]}")

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = IPATokenizer()
        # style vectors are packed per token-length — index by len(tokens)
        self.speaker = np.array(voices[voice], dtype=np.float32)

        print(f"  tts: kokoro ({repo}), {threads} threads")

        if warmup:
            t0 = time.perf_counter()
            try:
                self("Warming up the speech engine.")
                print(f"  tts warm-up: {time.perf_counter() - t0:.2f}s (paid upfront)")
            except Exception as e:
                print(f"  tts warm-up failed: {e}")

    def __call__(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text)
        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.session.run(
            None,
            {
                "tokens": [[0, *tokens, 0]],
                "style": self.speaker[len(tokens)],
                "speed": np.ones(1, dtype=np.float32) * self.speed,
            },
        )
        return np.asarray(out[0], dtype=np.float32).flatten()
