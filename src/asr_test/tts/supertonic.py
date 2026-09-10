from __future__ import annotations

import json
import re
from pathlib import Path
from unicodedata import normalize

import numpy as np
import onnxruntime as ort

from ..interfaces.tts import TtsBase

# src/asr_test/tts/supertonic.py -> project root -> ./assets
# Populated by `git clone https://huggingface.co/Supertone/supertonic-3 assets`
# (same onnx/ + voice_styles/ layout as the Hub repo). Gitignored — see .gitignore.
_LOCAL_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

_VALID_VOICES = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")
_AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]

# Anything outside word chars / whitespace / apostrophe is stripped before
# synthesis — cheap insurance against unicode_indexer KeyErrors on symbols
# the model wasn't trained on (dashes, curly quotes, emoji, ...).
_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def _strip_punctuation(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text)).strip()


# ------------------------------ engine internals ------------------------------
# Ported from Supertone's reference ONNX pipeline: text -> duration predictor
# -> text encoder -> iterative (diffusion-style) vector estimator -> vocoder.

class _UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path) as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)

        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
            "☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        replacements = {
            "–": "-", "‑": "-", "—": "-", "_": " ",
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "´": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ",
            "#": " ", "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[♥☆♡©\\]", "", text)

        for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
            text = text.replace(k, v)

        for pat, rep in [(r" ,", ","), (r" \.", "."), (r" !", "!"), (r" \?", "?"),
                         (r" ;", ";"), (r" :", ":"), (r" '", "'")]:
            text = re.sub(pat, rep, text)

        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")

        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        if lang not in _AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        return f"<{lang}>" + text + f"</{lang}>"

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)

    def __call__(self, text_list: list[str], lang_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list, strict=True)]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, : len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        text_mask = _length_to_mask(text_ids_lengths)
        return text_ids, text_mask


class _Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class _TextToSpeechEngine:
    def __init__(self, cfgs: dict, text_processor: _UnicodeProcessor,
                 dp_ort: ort.InferenceSession, text_enc_ort: ort.InferenceSession,
                 vector_est_ort: ort.InferenceSession, vocoder_ort: ort.InferenceSession):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = _get_latent_mask(wav_lengths, self.base_chunk_size, self.chunk_compress_factor)
        return noisy_latent * latent_mask, latent_mask

    def _infer(self, text_list: list[str], lang_list: list[str], style: _Style,
               total_step: int, speed: float) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list, lang_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
        )
        xt, latent_mask = self._sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def __call__(self, text: str, lang: str, style: _Style, total_step: int,
                 speed: float, silence_duration: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        max_len = 120 if lang in ("ko", "ja") else 300
        text_list = _chunk_text(text, max_len=max_len)
        wav_cat = None
        dur_cat = None
        for chunk in text_list:
            wav, dur_onnx = self._infer([chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat, dur_cat = wav, dur_onnx
            else:
                silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur_onnx + silence_duration
        return wav_cat, dur_cat


def _length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def _get_latent_mask(wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return _length_to_mask(latent_lengths)


def _chunk_text(text: str, max_len: int = 300) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: list[str] = []
    pattern = (
        r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)"
        r"(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)"
        r"(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
    )
    for paragraph in paragraphs:
        sentences = re.split(pattern, paragraph)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_len:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
    return chunks


def _load_voice_style(path: str) -> _Style:
    with open(path) as f:
        raw = json.load(f)
    ttl_dims = raw["style_ttl"]["dims"]
    dp_dims = raw["style_dp"]["dims"]
    ttl = np.array(raw["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_dims[1], ttl_dims[2])
    dp = np.array(raw["style_dp"]["data"], dtype=np.float32).reshape(1, dp_dims[1], dp_dims[2])
    return _Style(ttl, dp)


# ------------------------------ public wrapper ------------------------------

class SupertonicTts(TtsBase):
    """Supertone's supertonic-3 ONNX pipeline (duration predictor -> text
    encoder -> iterative vector estimator -> vocoder). `total_steps`
    controls the diffusion-style refinement loop: more steps trade latency
    for quality.
    """

    VALID_VOICES = _VALID_VOICES

    @classmethod
    def list_voices(cls) -> list[str]:
        return list(cls.VALID_VOICES)

    def __init__(
        self,
        repo: str = "Supertone/supertonic-3",
        local_dir: str | None = None,
        voice: str = "F5",
        lang: str = "en",
        speed: float = 1.05,
        total_steps: int = 8,
        threads: int = 8,
        warmup: bool = True,
    ):
        import time as _time

        if voice not in _VALID_VOICES:
            raise ValueError(f"voice '{voice}' invalid; have {_VALID_VOICES}")

        if local_dir is None and (_LOCAL_ASSETS_DIR / "onnx" / "tts.json").exists():
            local_dir = str(_LOCAL_ASSETS_DIR)

        if local_dir is not None:
            # Same layout as the Hub repo (onnx/*.onnx + voice_styles/*.json),
            # just already on disk — skip the network round-trip entirely.
            base = Path(local_dir)
            dp_path = str(base / "onnx" / "duration_predictor.onnx")
            text_enc_path = str(base / "onnx" / "text_encoder.onnx")
            vector_est_path = str(base / "onnx" / "vector_estimator.onnx")
            vocoder_path = str(base / "onnx" / "vocoder.onnx")
            cfg_path = str(base / "onnx" / "tts.json")
            indexer_path = str(base / "onnx" / "unicode_indexer.json")
            style_path = str(base / "voice_styles" / f"{voice}.json")
        else:
            from huggingface_hub import hf_hub_download

            dp_path = hf_hub_download(repo, "onnx/duration_predictor.onnx")
            text_enc_path = hf_hub_download(repo, "onnx/text_encoder.onnx")
            vector_est_path = hf_hub_download(repo, "onnx/vector_estimator.onnx")
            vocoder_path = hf_hub_download(repo, "onnx/vocoder.onnx")
            cfg_path = hf_hub_download(repo, "onnx/tts.json")
            indexer_path = hf_hub_download(repo, "onnx/unicode_indexer.json")
            style_path = hf_hub_download(repo, f"voice_styles/{voice}.json")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Unlike Kokoro (one session), this pipeline is 4 separate sessions run
        # sequentially. ORT's intra-op threads busy-spin waiting for work by
        # default, so the 3 idle sessions' pools contend with the 1 active one
        # for CPU even though they're never doing real work concurrently.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]

        dp_ort = ort.InferenceSession(dp_path, sess_options=opts, providers=providers)
        text_enc_ort = ort.InferenceSession(text_enc_path, sess_options=opts, providers=providers)
        vector_est_ort = ort.InferenceSession(vector_est_path, sess_options=opts, providers=providers)
        vocoder_ort = ort.InferenceSession(vocoder_path, sess_options=opts, providers=providers)

        with open(cfg_path) as f:
            cfgs = json.load(f)
        text_processor = _UnicodeProcessor(indexer_path)

        self._engine = _TextToSpeechEngine(
            cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
        )
        self._style = _load_voice_style(style_path)
        self.sample_rate = self._engine.sample_rate
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.total_steps = total_steps

        source = local_dir or repo
        print(f"  tts: supertonic ({source}), voice={voice}, {threads} threads")

        if warmup:
            t0 = _time.perf_counter()
            try:
                self("Warming up the speech engine.")
                print(f"  tts warm-up: {_time.perf_counter() - t0:.2f}s (paid upfront)")
            except Exception as e:
                print(f"  tts warm-up failed: {e}")

    def __call__(self, text: str) -> np.ndarray:
        text = _strip_punctuation(text or "")
        if not text:
            return np.zeros(0, dtype=np.float32)
        wav, dur = self._engine(text, self.lang, self._style, self.total_steps, self.speed)
        usable = int(self.sample_rate * dur[0].item())
        return np.asarray(wav[0, :usable], dtype=np.float32)
