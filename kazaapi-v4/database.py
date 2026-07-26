"""
database.py
-----------
Thin async wrapper around Supabase (PostgREST) so the rest of the codebase
never talks to `supabase-py` directly. This gives us one place to:
  - reuse a single httpx connection pool (avoids reconnecting per-request,
    which matters when Render's free tier throttles CPU),
  - centralize error handling so a Supabase hiccup never 500s the whole app,
  - keep query logic testable/mockable.

We use the official `supabase` python package's async client
(`create_async_client`), which itself wraps `httpx.AsyncClient` and reuses
connections, satisfying SECTION 8's "Asynchronous I/O" requirement.

IMPORTANT: every public function below wraps its *entire* body -- including
the `get_client()` call -- in try/except. `get_client()` can itself raise
(missing env vars, a malformed key, a network blip while creating the
client), and if that happened outside a try/except it would propagate all
the way up through a router (or worse, through a BackgroundTask, where
FastAPI has nothing to catch it and it just gets logged as an unhandled
exception with no response to send). Catching it here means a misconfigured
or briefly-unreachable Supabase project degrades a single feature
gracefully instead of taking the whole request down, per SECTION 8's
"graceful fallbacks for every external API; log errors but never crash".
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase import AsyncClient, create_async_client

logger = logging.getLogger("kazaapi.database")

_client: Optional[AsyncClient] = None


async def get_client() -> AsyncClient:
    """Lazily create (once) and reuse a single Supabase async client.
    May raise — callers must call this from within their own try/except."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set as environment variables."
            )
        _client = await create_async_client(url, key)
        logger.info("Supabase async client initialized.")
    return _client


# ---------------------------------------------------------------------------
# temporary_chats
# ---------------------------------------------------------------------------

async def insert_chat_message(session_id: str, sender_role: str, content: str) -> dict[str, Any]:
    try:
        client = await get_client()
        res = await (
            client.table("temporary_chats")
            .insert({"session_id": session_id, "sender_role": sender_role, "content": content})
            .execute()
        )
        return res.data[0] if res.data else {}
    except Exception as exc:
        logger.error("insert_chat_message failed: %s", exc)
        return {}


async def get_recent_messages(session_id: str, limit: int = 30) -> list[dict[str, Any]]:
    try:
        client = await get_client()
        res = await (
            client.table("temporary_chats")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(reversed(res.data or []))
    except Exception as exc:
        logger.error("get_recent_messages failed: %s", exc)
        return []


async def get_last_message_time(session_id: str) -> Optional[datetime]:
    try:
        client = await get_client()
        res = await (
            client.table("temporary_chats")
            .select("created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            ts = res.data[0]["created_at"]
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as exc:
        logger.error("get_last_message_time failed: %s", exc)
    return None


async def clear_session(session_id: str) -> bool:
    try:
        client = await get_client()
        await client.table("temporary_chats").delete().eq("session_id", session_id).execute()
        return True
    except Exception as exc:
        logger.error("clear_session failed: %s", exc)
        return False


async def delete_messages_before(session_id: str, cutoff: datetime) -> int:
    try:
        client = await get_client()
        res = await (
            client.table("temporary_chats")
            .delete()
            .eq("session_id", session_id)
            .lt("created_at", cutoff.isoformat())
            .execute()
        )
        return len(res.data or [])
    except Exception as exc:
        logger.error("delete_messages_before failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# long_term_memories
# ---------------------------------------------------------------------------

async def upsert_long_term_summary(session_id: str, summary: str) -> None:
    try:
        client = await get_client()
        await (
            client.table("long_term_memories")
            .upsert({"session_id": session_id, "summary": summary,
                     "updated_at": datetime.now(timezone.utc).isoformat()},
                    on_conflict="session_id")
            .execute()
        )
    except Exception as exc:
        logger.error("upsert_long_term_summary failed: %s", exc)


async def get_long_term_summary(session_id: str) -> Optional[str]:
    try:
        client = await get_client()
        res = await (
            client.table("long_term_memories")
            .select("summary")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["summary"]
    except Exception as exc:
        logger.error("get_long_term_summary failed: %s", exc)
    return None


async def purge_old_records(days: int = 7) -> None:
    """Backstop for the SQL cron/trigger in schema.sql — safe to call from
    a BackgroundTask too, in case pg_cron isn't available on the Supabase
    free tier project."""
    try:
        client = await get_client()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        await client.table("temporary_chats").delete().lt("created_at", cutoff).execute()
        await client.table("long_term_memories").delete().lt("updated_at", cutoff).execute()
    except Exception as exc:
        logger.error("purge_old_records failed: %s", exc)


# ---------------------------------------------------------------------------
# user_preferences
# ---------------------------------------------------------------------------

async def get_preferences(user_key: str = "kaza") -> Optional[dict[str, Any]]:
    try:
        client = await get_client()
        res = await (
            client.table("user_preferences")
            .select("*")
            .eq("user_key", user_key)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.error("get_preferences failed: %s", exc)
        return None


async def upsert_preferences(user_key: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = {"user_key": user_key, **data,
               "updated_at": datetime.now(timezone.utc).isoformat()}
    try:
        client = await get_client()
        res = await (
            client.table("user_preferences")
            .upsert(payload, on_conflict="user_key")
            .execute()
        )
        return res.data[0] if res.data else payload
    except Exception as exc:
        logger.error("upsert_preferences failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# flashcards
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Raised by the flashcard CRUD helpers so routers/tools.py can turn
    a genuine Supabase failure into a proper HTTP 502, instead of the
    silent best-effort fallback used for chat/memory (flashcards are
    explicit CRUD operations where the caller needs to know a write
    actually failed, rather than silently no-op'ing)."""


async def create_flashcard(row: dict[str, Any]) -> dict[str, Any]:
    try:
        client = await get_client()
        res = await client.table("flashcards").insert(row).execute()
        return res.data[0] if res.data else {}
    except Exception as exc:
        logger.error("create_flashcard failed: %s", exc)
        raise DatabaseError(str(exc)) from exc


async def list_flashcards(deck: Optional[str] = None, due_only: bool = False) -> list[dict[str, Any]]:
    try:
        client = await get_client()
        q = client.table("flashcards").select("*")
        if deck:
            q = q.eq("deck", deck)
        if due_only:
            q = q.lte("due_at", datetime.now(timezone.utc).isoformat())
        res = await q.order("due_at", desc=False).execute()
        return res.data or []
    except Exception as exc:
        logger.error("list_flashcards failed: %s", exc)
        return []


async def get_flashcard(card_id: str) -> Optional[dict[str, Any]]:
    try:
        client = await get_client()
        res = await client.table("flashcards").select("*").eq("id", card_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.error("get_flashcard failed: %s", exc)
        return None


async def update_flashcard(card_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    try:
        client = await get_client()
        res = await client.table("flashcards").update(patch).eq("id", card_id).execute()
        return res.data[0] if res.data else {}
    except Exception as exc:
        logger.error("update_flashcard failed: %s", exc)
        raise DatabaseError(str(exc)) from exc


async def delete_flashcard(card_id: str) -> bool:
    try:
        client = await get_client()
        await client.table("flashcards").delete().eq("id", card_id).execute()
        return True
    except Exception as exc:
        logger.error("delete_flashcard failed: %s", exc)
        raise DatabaseError(str(exc)) from exc
