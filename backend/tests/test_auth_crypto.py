import base64
import os

import pytest
from cryptography.exceptions import InvalidTag

from pos import config
from pos.auth.crypto import decrypt_payload, encrypt_payload


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())


def test_round_trips_a_payload():
    payload = {"OPENAI_API_KEY": "sk-test-123"}
    ciphertext, nonce = encrypt_payload(payload)
    assert decrypt_payload(ciphertext, nonce) == payload


def test_ciphertext_does_not_contain_the_plaintext():
    ciphertext, _ = encrypt_payload({"OPENAI_API_KEY": "sk-test-123"})
    assert b"sk-test-123" not in ciphertext


def test_each_encryption_uses_a_fresh_nonce():
    _, nonce_a = encrypt_payload({"k": "v"})
    _, nonce_b = encrypt_payload({"k": "v"})
    assert nonce_a != nonce_b
    assert len(nonce_a) == 12


def test_tampered_ciphertext_is_rejected():
    ciphertext, nonce = encrypt_payload({"k": "v"})
    tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]
    with pytest.raises(InvalidTag):
        decrypt_payload(tampered, nonce)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "ENCRYPTION_KEY", "")
    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
        encrypt_payload({"k": "v"})
