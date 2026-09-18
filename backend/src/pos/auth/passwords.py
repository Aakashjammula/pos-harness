"""Argon2id password hashing. OWASP's Password Storage Cheat Sheet ranks
Argon2id above bcrypt: memory-hard, so GPU/ASIC cracking gains less, and
time and memory cost are tunable independently. argon2-cffi's defaults
already exceed OWASP's floor (19 MiB / 2 iterations / 1 parallelism)."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (Argon2Error, ValueError):
        return False
