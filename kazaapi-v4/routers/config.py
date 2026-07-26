"""
routers/config.py
------------------
SECTION 5: user preference / config management. Every route here requires
a valid kaza token — strangers can't view or change anything, and even
kaza never gets raw API key values back in a GET (only booleans indicating
whether each key is configured).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

import database
from models.schemas import ConfigResponse, PersonalitySliders, UserPreferences
from utils.auth import require_kaza
from utils.crypto import encrypt_dict

router = APIRouter(
    prefix="/api/v1/config",
    tags=["config"],
    dependencies=[Depends(require_kaza)],
)

KNOWN_KEY_NAMES = ("GEMINI_KEY", "GROQ_KEY", "OPENROUTER_KEY", "SERPAPI_KEY")


@router.get("", response_model=ConfigResponse)
async def get_config():
    row = await database.get_preferences("kaza") or {}
    personality_data = row.get("personality") or {}
    stored_keys = (row.get("api_keys") or {})  # values are Fernet ciphertext, never decrypted here
    return ConfigResponse(
        personality=PersonalitySliders(**personality_data),
        tool_toggles=row.get("tool_toggles") or {},
        api_keys_configured={name: name in stored_keys for name in KNOWN_KEY_NAMES},
    )


@router.put("", response_model=ConfigResponse)
async def update_config(prefs: UserPreferences):
    existing = await database.get_preferences("kaza") or {}
    existing_keys = existing.get("api_keys") or {}

    new_encrypted_keys = encrypt_dict(prefs.api_keys) if prefs.api_keys else {}
    merged_keys = {**existing_keys, **new_encrypted_keys}

    payload = {
        "personality": prefs.personality.model_dump(),
        "tool_toggles": prefs.tool_toggles,
        "api_keys": merged_keys,
    }
    await database.upsert_preferences("kaza", payload)

    return ConfigResponse(
        personality=prefs.personality,
        tool_toggles=prefs.tool_toggles,
        api_keys_configured={name: name in merged_keys for name in KNOWN_KEY_NAMES},
    )
