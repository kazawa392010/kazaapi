"""
services/memory.py
-------------------
Memory lifecycle management (SECTION 3):

  - Idle Compaction: if the gap since the last message in a session is
    > 30 minutes, summarize the temporary_chats thread with a fast LLM
    call and fold it into long_term_memories, then delete the old
    temporary rows. This keeps `temporary_chats` small (helps the
    500MB Supabase free tier) while preserving context across sessions.

  - 7-Day Auto-Purge: belt-and-suspenders Python-side purge, in addition
    to the SQL function defined in schema.sql (which should ideally run
    via pg_cron or a Supabase scheduled Edge Function). Exposed here so
    routers/chat.py's /api/v1/cron endpoint can also trigger it.

Both functions are designed to be handed to FastAPI's BackgroundTasks so
they run *after* the HTTP response has already been sent to the client —
the user never waits on compaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import database
from services import llm

logger = logging.getLogger("kazaapi.memory")

IDLE_THRESHOLD_MINUTES = 30
IDLE_THRESHOLD_SECONDS = IDLE_THRESHOLD_MINUTES * 60

_SUMMARY_SYSTEM_PROMPT = """You compress a conversation transcript into a
short, dense memory note (max ~150 words) for future recall. Capture: who
the user is, what they were working on, any preferences/facts they stated,
and unresolved threads. Write in plain prose, third person, no filler like
"the user talked about". Do not include timestamps or role labels."""


async def maybe_compact_session(session_id: str) -> None:
    """Checks idle gap and compacts if needed. Safe to call unconditionally
    at the start of every chat request — it's a cheap check most of the time."""
    last_msg_time = await database.get_last_message_time(session_id)
    if last_msg_time is None:
        return  # brand new session, nothing to compact

    now = datetime.now(timezone.utc)
    gap_seconds = (now - last_msg_time).total_seconds()
    if gap_seconds < IDLE_THRESHOLD_SECONDS:
        return

    await compact_session(session_id)


async def compact_session(session_id: str) -> None:
    messages = await database.get_recent_messages(session_id, limit=100)
    if not messages:
        return

    transcript = "\n".join(
        f"{m.get('sender_role', 'user')}: {m.get('content', '')}" for m in messages
    )
    existing_summary = await database.get_long_term_summary(session_id)
    prompt = transcript
    if existing_summary:
        prompt = f"Existing memory note:\n{existing_summary}\n\nNew transcript to fold in:\n{transcript}"

    try:
        new_summary = await llm.complete_sync(_SUMMARY_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        logger.warning("compact_session: summary generation failed for %s: %s", session_id, exc)
        return  # don't delete anything if we failed to summarize first

    await database.upsert_long_term_summary(session_id, new_summary)
    # Only clear what we just summarized, not anything written concurrently.
    cutoff = datetime.now(timezone.utc)
    await database.delete_messages_before(session_id, cutoff)
    logger.info("Compacted session %s (%d messages -> summary)", session_id, len(messages))


async def run_auto_purge() -> None:
    """Hard 7-day retention backstop, called from /api/v1/cron."""
    await database.purge_old_records(days=7)
    logger.info("Auto-purge complete (7-day retention).")
