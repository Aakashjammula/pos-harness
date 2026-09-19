"""Shared, engine-agnostic settings for the voice agent.

Settings specific to one VAD/STT/LLM/TTS implementation (model repo, API
base URL, voice name, ...) live as constructor defaults on that
implementation instead, so swapping an engine doesn't require touching
this file.
"""

from __future__ import annotations

import os
import re

MIC_RATE = 16000
FRAME = 512
OUT_BLOCK = 1024

BARGE_IN_FRAMES = 3
BARGE_IN_GRACE_MS = 250

TRIGGER_LOOKAHEAD_WORDS = 2         # only matters if --trigger-word is set: how many leading
                                     # words (beyond the trigger phrase itself) to tolerate before
                                     # it — covers STT-transcribed filler like "uh"/"hey"/"okay"
                                     # picked up before the real trigger word. Kept small on
                                     # purpose: a larger window increases false triggers on
                                     # sentences that merely mention the word (e.g. "tell me about
                                     # the computer" has 4 words before "computer" — tune up only
                                     # if you're seeing genuine commands dropped for too much
                                     # leading filler, not just to be safe.

MIN_SILENCE_MS = 1200              # VAD can't tell a mid-sentence pause from turn-end; bias toward not cutting off.
                                    # Was 900 — too short: a normal hesitation pause ("Uh... hello...")
                                    # closed the segment early, fragmenting one utterance into several
                                    # short ones. Confirmed the STT model occasionally returns an empty
                                    # transcript for short (~2-3s) segments even when they contain real
                                    # speech (a longer, complete segment didn't fail the same way) — so
                                    # fragmenting increases exposure to that failure mode. Trade-off:
                                    # +300ms added latency after the user stops talking, in exchange for
                                    # fewer silently-dropped fragments. Tune down if the delay feels bad.
SPEECH_PAD_MS = 300
MIN_SPEECH_SEC = 0.4
MAX_SEGMENT_SEC = 20.0

# TTS synth cost scales with text length, so keep the opening fragment short.
FIRST_CHUNK_CHARS = 25
MAX_CHUNK_CHARS = 140
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

VERBOSE_TIMING = True
REALTIME_LOG = True               # per-stage trigger log: VAD / STT / LLM / TTS

HISTORY_TURNS = 3   # how many prior user/assistant turn-pairs to include as LLM
                     # context. Previously owned by the LLM class itself;
                     # moved to Agent since the LLM engine is now stateless and
                     # (in server mode) shared across concurrent sessions.

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos"
)   # Postgres DSN for session/turn history — see storage.py
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "10"))

JWT_SECRET = os.environ.get("JWT_SECRET", "")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "")

# Where the magic-link email points -- the frontend, not this backend, since
# the browser needs to land on a page that calls back into /auth/magic-link/verify
# with cookies attached (a direct link to the backend would cross origins).
FRONTEND_URL = os.environ.get("FRONTEND_URL", CORS_ORIGINS[0] if CORS_ORIGINS else "http://localhost:3000")

# fastapi-mail's ConnectionConfig wants MAIL_STARTTLS/MAIL_SSL_TLS (current
# versions); kept as MAIL_TLS/MAIL_SSL here to match how this deployment's
# env is already named, and translated when building ConnectionConfig.
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "")
MAIL_SERVER = os.environ.get("MAIL_SERVER", "")
MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
MAIL_TLS = os.environ.get("MAIL_TLS", "true").lower() == "true"
MAIL_SSL = os.environ.get("MAIL_SSL", "false").lower() == "true"
