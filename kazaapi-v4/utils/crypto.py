"""
utils/crypto.py
----------------
Symmetric encryption for anything sensitive we persist in Supabase
(user-supplied provider API keys in `user_preferences`).

We use `cryptography`'s Fernet (AES-128-CBC + HMAC, authenticated) rather than
rolling our own. Fernet tokens are self-contained (they embed a timestamp and
IV) so we don't need extra columns to store nonces/IVs ourselves — this keeps
the Supabase schema smaller, which matters on the 500MB free tier.

MASTER_KEY must be a urlsafe-base64-encoded 32 byte key, e.g. generated with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Set it as the MASTER_KEY environment variable on Render. Never commit it.
"""

from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken


class CryptoError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    key = os.environ.get("MASTER_KEY")
    if not key:
        raise CryptoError(
            "MASTER_KEY environment variable is not set. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"` and set it on Render."
        )
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as exc:  # invalid key format
        raise CryptoError(f"MASTER_KEY is not a valid Fernet key: {exc}") from exc


def encrypt_str(plaintext: str) -> str:
    """Encrypt a plaintext string, returning a urlsafe base64 token (str)."""
    if plaintext is None:
        return plaintext
    f = _get_fernet()
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_str(token: str) -> str:
    """Decrypt a token produced by encrypt_str. Raises CryptoError on failure."""
    if token is None:
        return token
    f = _get_fernet()
    try:
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise CryptoError("Could not decrypt value: invalid token or wrong MASTER_KEY") from exc


def encrypt_dict(values: dict[str, str]) -> dict[str, str]:
    """Encrypt every value in a flat dict (used for the api_keys JSONB blob)."""
    return {k: encrypt_str(v) for k, v in values.items() if v}


def decrypt_dict(values: dict[str, str]) -> dict[str, str]:
    out = {}
    for k, v in (values or {}).items():
        try:
            out[k] = decrypt_str(v)
        except CryptoError:
            # Corrupt/legacy value — skip rather than crash the whole config load.
            continue
    return out
