"""
utils/auth.py
--------------
RBAC enforcement (SECTION 1 & 5).

Important design note: the spec's `role: kaza | stranger` can't just be a
field the client sets on the request body — anyone could type
`"role": "kaza"` in their JSON and get full access, which defeats the
entire point of RBAC. Instead:

  - The real owner authenticates by sending header `X-Kaza-Token: <secret>`
    matching the KAZA_ACCESS_TOKEN environment variable.
  - `resolve_role()` downgrades any claimed role to `stranger` unless that
    token is present and correct — callers should always use the *returned*
    role, never trust `ChatRequest.role` directly.
  - `require_kaza()` is a FastAPI dependency for endpoints that must hard-
    reject strangers entirely (tool execution, config writes, Keep mock),
    per SECTION 1 ("stranger ... no tool execution, no memory write") and
    SECTION 5 ("only role == 'kaza' can modify config").

Set KAZA_ACCESS_TOKEN to a long random string on Render; never commit it.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

from models.schemas import Role


def _expected_token() -> str | None:
    return os.environ.get("KAZA_ACCESS_TOKEN")


def _token_matches(provided: str | None) -> bool:
    expected = _expected_token()
    if not expected or not provided:
        return False
    # constant-time comparison to avoid timing side-channels on the secret
    return secrets.compare_digest(provided, expected)


def resolve_role(claimed_role: Role, x_kaza_token: str | None) -> Role:
    """Given what the client *claims* and the token they *actually* sent,
    return the role that should really be trusted."""
    if claimed_role == Role.kaza and _token_matches(x_kaza_token):
        return Role.kaza
    return Role.stranger


async def require_kaza(x_kaza_token: str | None = Header(default=None)) -> Role:
    """FastAPI dependency: 403s unless a valid kaza token is presented.
    Use on any route that SECTION 1/5 restrict to role == 'kaza'."""
    if not _token_matches(x_kaza_token):
        raise HTTPException(status_code=403, detail="This action requires kaza access (missing/invalid X-Kaza-Token).")
    return Role.kaza
