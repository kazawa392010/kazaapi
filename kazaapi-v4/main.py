"""
main.py
-------
KazaAPI v4 entrypoint.

Wires together:
  - CORS (frontend is hosted separately on Vercel/GitHub Pages)
  - routers: chat, tools, config
  - GET /api/v1/cron — the "Proactive / Idle Engine" (SECTION 4.C):
    an external uptime pinger (e.g. UptimeRobot) hits this on a schedule.
    Each ping (a) runs the 7-day memory auto-purge and (b) with a small
    probability, queues a background self-learning task (scrape a random
    topic and summarize it into long-term memory) — all as BackgroundTasks
    so the HTTP response returns instantly and the pinger never times out.
  - GET /healthz — cheap liveness check for Render + the frontend's health badge.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import random

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, config, tools
from services import agent, llm, memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kazaapi.main")

app = FastAPI(
    title="KazaAPI v4",
    description="Backend-centric, full-stack AI study assistant (Uri) for students.",
    version="4.0.0",
)

# CORS: the frontend (index.html) is a fully decoupled static client hosted
# elsewhere. Configure ALLOWED_ORIGINS as a comma-separated env var in
# production (e.g. "https://kazaapi.vercel.app"); defaults to "*" for local dev.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if _allowed_origins_env.strip() == "*"
    else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],  # can't combine "*" with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(tools.router)
app.include_router(config.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "kazaapi-v4"}


# ---------------------------------------------------------------------------
# Proactive / Idle Engine
# ---------------------------------------------------------------------------

PROACTIVE_PROBABILITY = 0.15  # small chance per cron ping, per SECTION 4.C
SELF_LEARN_TOPICS = [
    "spaced repetition study techniques",
    "interesting number theory facts",
    "how photosynthesis works at the molecular level",
    "the history of the Vietnamese language",
    "common Python performance pitfalls",
    "how neural networks backpropagate",
]


async def _self_learning_task() -> None:
    """Scrapes a random topic (via a DuckDuckGo lite search -> first result
    -> Jina Reader) and folds a summary into a shared 'uri-self-learning'
    long-term memory bucket, so Uri can casually bring it up later."""
    from services.search import web_search  # local import avoids a cold-start cost most requests never pay

    topic = random.choice(SELF_LEARN_TOPICS)
    try:
        results = await web_search(topic, max_results=1)
        if not results:
            return
        markdown, _ = await agent.scrape_url(results[0].url)
        summary = await llm.complete_sync(
            "Summarize this page in 2-3 sentences for a curious study assistant's memory.",
            markdown[:4000],
        )
        note = f"(self-learned about '{topic}'): {summary}"
        await _append_self_learning_note(note)
        logger.info("Self-learning task completed for topic: %s", topic)
    except Exception as exc:
        logger.warning("Self-learning task failed: %s", exc)


async def _append_self_learning_note(note: str) -> None:
    import database
    existing = await database.get_long_term_summary("uri-self-learning") or ""
    updated = (existing + "\n" + note).strip()[-3000:]  # keep it bounded
    await database.upsert_long_term_summary("uri-self-learning", updated)


@app.get("/api/v1/cron")
async def cron(background_tasks: BackgroundTasks):
    """External pinger endpoint (e.g. UptimeRobot every 10-14 min) that:
      1. keeps the free Render instance from spinning down on idle, and
      2. drives the idle/proactive engine.
    Everything heavy runs as a BackgroundTask so this returns in milliseconds.
    """
    background_tasks.add_task(memory.run_auto_purge)

    triggered_self_learning = False
    if random.random() < PROACTIVE_PROBABILITY:
        background_tasks.add_task(_self_learning_task)
        triggered_self_learning = True

    return {"pinged": True, "self_learning_queued": triggered_self_learning}
