"""AES-256-GCM envelope encryption for stored provider credentials.

GCM needs a unique nonce per encryption under one key; the nonce isn't
secret and is stored alongside the ciphertext. Losing ENCRYPTION_KEY
makes every stored credential unrecoverable -- back it up like any other
production secret."""

from __future__ import annotations

import base64
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pos import config

_NONCE_BYTES = 12


def _key() -> bytes:
    if not config.ENCRYPTION_KEY:
        raise RuntimeError("ENCRYPTION_KEY is not set -- cannot store provider credentials")
    key = base64.b64decode(config.ENCRYPTION_KEY)
    if len(key) != 32:
        raise RuntimeError("ENCRYPTION_KEY must decode to exactly 32 bytes (AES-256)")
    return key


def encrypt_payload(payload: dict) -> tuple[bytes, bytes]:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_key()).encrypt(nonce, json.dumps(payload).encode(), None)
    return ciphertext, nonce


def decrypt_payload(ciphertext: bytes, nonce: bytes) -> dict:
    return json.loads(AESGCM(_key()).decrypt(nonce, ciphertext, None))
