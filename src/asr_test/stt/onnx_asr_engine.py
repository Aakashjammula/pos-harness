from __future__ import annotations

import numpy as np

from ..interfaces.stt import SttBase


class OnnxAsrEngine(SttBase):
    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: str = "int8",
        warmup_sample_rate: int = 16000,
    ):
        import onnx_asr

        self.model = onnx_asr.load_model(model_name, quantization=quantization)
        try:
            self.model.recognize(np.zeros(warmup_sample_rate, dtype=np.float32))
        except Exception:
            pass

    def __call__(self, audio: np.ndarray) -> str:
        return (self.model.recognize(audio) or "").strip()
