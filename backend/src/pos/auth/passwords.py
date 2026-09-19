"""Argon2id password hashing. OWASP's Password Storage Cheat Sheet ranks
Argon2id above bcrypt: memory-hard, so GPU/ASIC cracking gains less, and
time and memory cost are tunable independently. argon2-cffi's defaults
already exceed OWASP's floor (19 MiB / 2 iterations / 1 parallelism)."""

from __future__ import annotations

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error

_hasher = PasswordHasher()

# Verified against when there is no real hash (unknown email, or an account with no
# password), so those logins cost the same as a wrong password and response time
# does not reveal which addresses have accounts. Its password is random and thrown away.
DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(24))


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (Argon2Error, ValueError):
        return False
