"""
routers/chat.py
----------------
Main chat surface: SSE streaming endpoint, history retrieval, and session
reset. This is the only router the frontend's primary chat UI talks to.

BUGFIXES applied here (see CHANGES.md):
  #3 "tính năng yêu cầu Uri làm nhưng không chạy" — this endpoint now
     actually calls services.llm.route_tool_call() + services.actions.
     dispatch_tool() so a message like "vẽ đồ thị y=x^2", "chạy code này",
     "tạo thẻ ghi nhớ"... really executes the corresponding SECTION 4 tool
     and feeds the real result back into Uri's answer, instead of Uri only
     ever being able to talk *about* tools in prose.
  #3 (continued) — provider API keys saved via PUT /api/v1/config
     (encrypted in Supabase) are now decrypted and actually passed to
     services.llm.stream_completion(); previously they were fetched but
     never forwarded, so only Render-environment-variable keys ever worked.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Header
from fastapi.responses import StreamingResponse

import database
from models.schemas import (
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResetResponse,
    EmotionVector,
    Mode,
)
from services import actions, llm, memory, search
from utils.auth import resolve_role
from utils.crypto import decrypt_dict

logger = logging.getLogger("kazaapi.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# In-memory per-session emotion state. Fine for a single Render worker;
# for multi-worker deployments this should move into user_preferences /
# a dedicated Supabase table so state is shared, but that's overkill for
# the free-tier single-instance deployment target here.
_emotion_state: dict[str, EmotionVector] = {}


@router.post("")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks,
               x_kaza_token: str | None = Header(default=None)):
    """SSE streaming chat endpoint.

    Emits Server-Sent Events of the shape:
        event: meta        -> {"role":..., "mode":..., "emotion": {...}}
        event: provider    -> {"name": ...}
        event: tool_call    -> {"name": ..., "args": {...}}
        event: tool_result   -> {"name": ..., "ok": bool, "payload": {...}}
        event: token        -> {"text": "..."}
        event: done         -> {"terminated": bool}
        event: error        -> {"message": "..."}

    (tool_call/tool_result are new — emitted only when the tool-routing
    classifier decides this message needs one of the SECTION 4 tools.)
    """
    # 0. Never trust req.role as-is — a client could just type "kaza" in the
    #    JSON body. Downgrade to 'stranger' unless the X-Kaza-Token header
    #    actually matches KAZA_ACCESS_TOKEN.
    role = resolve_role(req.role, x_kaza_token)

    # 1. Idle compaction check runs in the background so it never blocks
    #    this request's latency. Only meaningful for the owner's memory.
    if role.value == "kaza":
        background_tasks.add_task(memory.maybe_compact_session, req.session_id)

    # 2. Persist the incoming user message (RBAC: strangers still get their
    #    message stored short-term for context so the conversation reads
    #    coherently, but it is never promoted to long-term memory — that
    #    only happens via maybe_compact_session, which we only schedule for
    #    role == 'kaza' above).
    await database.insert_chat_message(req.session_id, "user", req.message)

    # 3. Update Uri's emotional vector.
    current_emotion = _emotion_state.get(req.session_id, EmotionVector())
    new_emotion = llm.update_emotion(current_emotion, req.message, role)
    _emotion_state[req.session_id] = new_emotion
    terminated = llm.should_terminate(new_emotion)

    # 4. Owner config: personality (kaza only) + provider API keys (BUGFIX:
    #    api_keys configured via PUT /api/v1/config are decrypted here and
    #    threaded all the way through to llm.stream_completion below,
    #    instead of being fetched-but-ignored like before). Every chat,
    #    regardless of who's talking, needs a working LLM provider to
    #    respond at all, so this is unconditional — not role-gated.
    prefs = await database.get_preferences("kaza") or {}
    personality = prefs.get("personality") if role.value == "kaza" else None
    configured_keys = decrypt_dict(prefs.get("api_keys") or {})

    # 5. BUGFIX #3: ask a fast classification pass whether this message
    #    actually calls for one of the SECTION 4 backend tools (in
    #    whatever language the user wrote it), and if so, run it for real.
    tool_call = None
    tool_outcome = None
    if not terminated:
        tool_call = await llm.route_tool_call(req.message, role, configured_keys)
        if tool_call:
            tool_outcome = await actions.dispatch_tool(tool_call["tool"], tool_call["args"])

    # 6. Dual-speed routing: infer casual vs task mode if not explicit. A
    #    successful (non-search) tool call is itself strong evidence of
    #    task-mode intent, so it takes priority over the plain keyword
    #    heuristic (which is still used as a fallback for messages the
    #    classifier decided needed no tool).
    if req.mode:
        mode = req.mode
    elif tool_call and tool_call["tool"] != "web_search":
        mode = Mode.task
    else:
        mode = Mode.task if llm.should_use_task_mode(req.message) else Mode.casual

    # 7. Context block for the final answer: prefer the real tool result;
    #    fall back to the old heuristic-triggered search pass only if the
    #    classifier didn't already run one (safety net, not the primary path).
    search_context = ""
    if tool_outcome is not None:
        if tool_outcome.ok:
            search_context = (
                f"\nYou just used the '{tool_call['tool']}' tool successfully. Use this result "
                f"in your reply — the user already sees any attached image/data directly, so "
                f"narrate around it naturally instead of repeating raw data back verbatim:\n"
                f"{tool_outcome.summary}\n"
            )
        else:
            search_context = (
                f"\nYou attempted to use the '{tool_call['tool']}' tool but it failed: "
                f"{tool_outcome.summary}\nBriefly explain the problem to the user in your own words "
                f"(don't pretend it worked).\n"
            )
    elif search.looks_like_factual_query(req.message):
        results = await search.web_search(req.message, max_results=5)
        search_context = search.format_results_for_prompt(results)

    long_term_summary = await database.get_long_term_summary(req.session_id) if role.value == "kaza" else None

    system_prompt = llm.build_system_prompt(
        role=role,
        mode=mode,
        emotion=new_emotion,
        long_term_summary=long_term_summary,
        personality=personality,
    )
    if search_context:
        system_prompt += f"\n{search_context}\n"

    async def event_stream():
        meta = {
            "role": role.value,
            "mode": mode.value,
            "emotion": new_emotion.model_dump(),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        if tool_call:
            yield f"event: tool_call\ndata: {json.dumps({'name': tool_call['tool'], 'args': tool_call['args']})}\n\n"
        if tool_outcome is not None:
            tool_event = {"name": tool_call["tool"], "ok": tool_outcome.ok, "payload": tool_outcome.payload}
            yield f"event: tool_result\ndata: {json.dumps(tool_event)}\n\n"

        if terminated:
            # Uri abruptly ends the exchange per SECTION 1.
            closing = "...yeah, I'm done for now. Come back when you're not like this. 🙄"
            yield f"event: token\ndata: {json.dumps({'text': closing})}\n\n"
            await database.insert_chat_message(req.session_id, "assistant", closing)
            yield f"event: done\ndata: {json.dumps({'terminated': True})}\n\n"
            return

        full_reply_parts: list[str] = []
        try:
            async for event in llm.stream_completion(system_prompt, req.message, configured_keys):
                if event["type"] == "provider":
                    yield f"event: provider\ndata: {json.dumps({'name': event['name']})}\n\n"
                elif event["type"] == "token":
                    full_reply_parts.append(event["text"])
                    yield f"event: token\ndata: {json.dumps({'text': event['text']})}\n\n"
                elif event["type"] == "error":
                    yield f"event: error\ndata: {json.dumps({'message': event['message']})}\n\n"
        except Exception as exc:
            logger.exception("chat stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

        full_reply = "".join(full_reply_parts)
        if full_reply:
            await database.insert_chat_message(req.session_id, "assistant", full_reply)

        yield f"event: done\ndata: {json.dumps({'terminated': False})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens stream promptly
        },
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_history(session_id: str, limit: int = 30):
    rows = await database.get_recent_messages(session_id, limit=limit)
    summary = await database.get_long_term_summary(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatHistoryItem(**row) for row in rows],
        long_term_summary=summary,
    )


@router.post("/reset", response_model=ChatResetResponse)
async def reset_session(session_id: str):
    ok = await database.clear_session(session_id)
    _emotion_state.pop(session_id, None)
    return ChatResetResponse(session_id=session_id, cleared=ok)
