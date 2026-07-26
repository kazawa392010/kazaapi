"""
services/agent.py
------------------
Two-stage scraping + form-submission agent (SECTION 4.A).

Stage 1 (scrape): fetch clean Markdown for any URL via Jina Reader
(https://r.jina.ai/<URL>) — free, no API key, and does the messy HTML->text
work for us so we don't need a headless browser (which would blow the
512MB RAM budget on Render's free tier).

Stage 2 (form submit): 
  a. scrape the target page,
  b. ask the LLM to extract the form's fields as structured JSON,
  c. diff against `known_values` the caller already has; if fields are
     still missing, return them so the chat layer can ask the user,
  d. once all required fields are known, POST the form with httpx and run
     the response page back through Jina + the LLM to produce a short
     summary (e.g. "You scored 82/100 on the English placement test").

Caching: scraped markdown is cached in-process for the lifetime of the
worker (functools.lru_cache doesn't support TTL natively, so we use a tiny
manual dict with timestamps) to avoid re-fetching the same URL repeatedly
within a session, per SECTION 8's caching guidance.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import httpx

from models.schemas import FormField
from services import llm

logger = logging.getLogger("kazaapi.agent")

SCRAPE_TIMEOUT = httpx.Timeout(30.0, connect=10.0)  # SECTION 8: 30s for scraping
JINA_READER_BASE = "https://r.jina.ai/"

_CACHE_TTL_SECONDS = 300
_scrape_cache: dict[str, tuple[float, str]] = {}

MAX_MARKDOWN_CHARS = 12000  # keep prompts (and RAM) bounded


async def scrape_url(url: str, use_cache: bool = True) -> tuple[str, bool]:
    """Returns (markdown, truncated). Cached in-memory for _CACHE_TTL_SECONDS."""
    now = time.time()
    if use_cache and url in _scrape_cache:
        ts, cached = _scrape_cache[url]
        if now - ts < _CACHE_TTL_SECONDS:
            truncated = len(cached) >= MAX_MARKDOWN_CHARS
            return cached[:MAX_MARKDOWN_CHARS], truncated

    reader_url = JINA_READER_BASE + url
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(reader_url, headers={"X-Return-Format": "markdown"})
        resp.raise_for_status()
        markdown = resp.text

    _scrape_cache[url] = (now, markdown)
    truncated = len(markdown) > MAX_MARKDOWN_CHARS
    return markdown[:MAX_MARKDOWN_CHARS], truncated


# ---------------------------------------------------------------------------
# Form field extraction (LLM pass #1)
# ---------------------------------------------------------------------------

_FIELD_EXTRACTION_SYSTEM_PROMPT = """You extract HTML form fields from a page's
Markdown/text content. Respond with ONLY a JSON array (no prose, no markdown
fences) of objects shaped like:
[{"name": "student_id", "label": "Student ID", "field_type": "text", "required": true}]
field_type must be one of: text, email, number, password, select, date, checkbox.
If you cannot find a real submittable form, respond with []."""


async def extract_form_fields(markdown: str) -> list[FormField]:
    prompt = f"Page content:\n{markdown[:6000]}\n\nExtract the form fields as JSON."
    try:
        raw = await llm.complete_sync(_FIELD_EXTRACTION_SYSTEM_PROMPT, prompt)
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        data = json.loads(raw)
        return [FormField(**item) for item in data]
    except Exception as exc:
        logger.warning("extract_form_fields failed: %s", exc)
        return []


def diff_missing_fields(fields: list[FormField], known_values: dict[str, str]) -> list[FormField]:
    return [f for f in fields if f.required and not known_values.get(f.name)]


# ---------------------------------------------------------------------------
# Form submission + result parsing (LLM pass #2)
# ---------------------------------------------------------------------------

_RESULT_SUMMARY_SYSTEM_PROMPT = """You are summarizing the result page returned
after submitting a form (e.g. a school portal score/grade page). In 2-4
sentences, extract the key outcome (score, grade, status, any listed
feedback). If nothing meaningful is present, say so plainly. Do not invent
numbers that are not in the text."""


async def submit_form(url: str, form_action_url: str, known_values: dict[str, str]) -> dict[str, Any]:
    """POSTs known_values to form_action_url and returns a parsed summary.

    form_action_url should already be resolved to an absolute URL by the
    caller (routers/tools.py), since Jina's markdown output doesn't reliably
    preserve the raw <form action=...> attribute — for real deployments you
    may want to fetch the raw HTML directly (not through Jina) just to
    resolve the action URL, then use Jina only for reading the result.
    """
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        resp = await client.post(form_action_url, data=known_values)
        resp.raise_for_status()
        result_html_excerpt = resp.text[:4000]

    # Run the raw response back through Jina for clean text, falling back to
    # the raw excerpt if that fails (e.g. Jina can't reach a POST result page
    # since it only re-fetches by GET — in that case we just summarize the
    # raw HTML excerpt directly).
    try:
        result_markdown, _ = await scrape_url(url, use_cache=False)
    except Exception:
        result_markdown = result_html_excerpt

    try:
        summary = await llm.complete_sync(_RESULT_SUMMARY_SYSTEM_PROMPT, result_markdown[:4000])
    except Exception as exc:
        logger.warning("submit_form summary failed: %s", exc)
        summary = None

    return {
        "status": "submitted",
        "summary": summary,
        "raw_excerpt": result_html_excerpt[:1000],
    }
