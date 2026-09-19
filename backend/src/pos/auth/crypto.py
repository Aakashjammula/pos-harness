"""AES-256-GCM envelope encryption for stored provider credentials.

GCM needs a unique nonce per encryption under one key; the nonce isn't
secret and is stored alongside the ciphertext. Losing ENCRYPTION_KEY
makes every stored credential unrecoverable -- back it up like any other
production secret.

Each ciphertext is bound to its owner with GCM's associated data (the caller
passes "<user id>:<provider>"): a row copied by someone with database write
access into another user's or provider's slot fails to decrypt instead of
handing the credential over.

Rotating the key: put the new key in ENCRYPTION_KEY and the old one(s) in
ENCRYPTION_KEY_PREVIOUS (comma-separated). Old ciphertexts still decrypt and are
re-encrypted under the current key the next time they are read; once nothing
needs the old key any more it can be removed."""

from __future__ import annotations

import base64
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pos import config

_NONCE_BYTES = 12


def _decode(key_b64: str) -> bytes:
    key = base64.b64decode(key_b64)
    if len(key) != 32:
        raise RuntimeError("an encryption key must decode to exactly 32 bytes (AES-256)")
    return key


def _key() -> bytes:
    if not config.ENCRYPTION_KEY:
        raise RuntimeError("ENCRYPTION_KEY is not set -- cannot store provider credentials")
    return _decode(config.ENCRYPTION_KEY)


def _previous_keys() -> list[bytes]:
    return [_decode(k.strip()) for k in config.ENCRYPTION_KEY_PREVIOUS.split(",") if k.strip()]


def encrypt_payload(payload: dict, aad: bytes | None = None) -> tuple[bytes, bytes]:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(_key()).encrypt(nonce, json.dumps(payload).encode(), aad)
    return ciphertext, nonce


def decrypt_payload_ex(ciphertext: bytes, nonce: bytes, aad: bytes | None = None) -> tuple[dict, bool]:
    """(payload, needs_upgrade). needs_upgrade is True when the row was written
    without associated data, or under a previous key, so the caller should
    re-encrypt it in the current format. Raises InvalidTag if nothing fits."""
    current = _key()
    attempts = [(current, aad, False)]
    if aad is not None:
        attempts.append((current, None, True))                       # the format before associated data
    for previous in _previous_keys():
        attempts.append((previous, aad, True))
        if aad is not None:
            attempts.append((previous, None, True))
    for key, associated, upgrade in attempts:
        try:
            return json.loads(AESGCM(key).decrypt(nonce, ciphertext, associated)), upgrade
        except InvalidTag:
            continue
    raise InvalidTag()


def decrypt_payload(ciphertext: bytes, nonce: bytes, aad: bytes | None = None) -> dict:
    return decrypt_payload_ex(ciphertext, nonce, aad)[0]
