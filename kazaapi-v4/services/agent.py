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

BUGFIXES applied here (see CHANGES.md #1):
  - Jina Reader is a shared free public service and does occasionally
    rate-limit or time out; previously any such failure made scrape_url()
    raise straight out of the function, which every caller (chat tool
    calls, /api/v1/tools/scrape, submit_form, self-learning) surfaced as
    "can't read that page". A direct-fetch-and-strip-tags fallback (using
    the already-installed selectolax parser) now kicks in so a Jina hiccup
    degrades to "plainer" text instead of a hard failure.
  - Added resolve_form_action(): Jina's markdown conversion drops the raw
    <form action=...> HTML attribute, so submit_form() was always POSTing
    back to the *page's own URL* instead of the form's real target —
    meaning form submission essentially never worked against a real form.
    This fetches the raw HTML directly (not through Jina) just to resolve
    the actual action URL.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

from models.schemas import FormField
from services import llm

logger = logging.getLogger("kazaapi.agent")

SCRAPE_TIMEOUT = httpx.Timeout(30.0, connect=10.0)  # SECTION 8: 30s for scraping
JINA_READER_BASE = "https://r.jina.ai/"
_DIRECT_FETCH_UA = "Mozilla/5.0 (compatible; KazaAPI/4.0; +https://kazaapi.onrender.com)"

_CACHE_TTL_SECONDS = 300
_scrape_cache: dict[str, tuple[float, str]] = {}

MAX_MARKDOWN_CHARS = 12000  # keep prompts (and RAM) bounded


async def _fetch_via_jina(url: str) -> str:
    reader_url = JINA_READER_BASE + url
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(reader_url, headers={"X-Return-Format": "markdown"})
        resp.raise_for_status()
        return resp.text


async def _fetch_direct_fallback(url: str) -> str:
    """Best-effort plain-text fallback when Jina Reader is unavailable:
    fetch the raw HTML ourselves and strip it down to visible text with
    selectolax (already a dependency, used for services/search.py too).
    Won't be as clean as Jina's markdown, but keeps scraping working
    instead of failing outright."""
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _DIRECT_FETCH_UA})
        resp.raise_for_status()
        html = resp.text
    tree = HTMLParser(html)
    for tag in tree.css("script, style, noscript"):
        tag.decompose()
    text = tree.body.text(separator="\n", strip=True) if tree.body else tree.text(separator="\n", strip=True)
    return text


async def scrape_url(url: str, use_cache: bool = True) -> tuple[str, bool]:
    """Returns (markdown, truncated). Cached in-memory for _CACHE_TTL_SECONDS."""
    now = time.time()
    if use_cache and url in _scrape_cache:
        ts, cached = _scrape_cache[url]
        if now - ts < _CACHE_TTL_SECONDS:
            truncated = len(cached) >= MAX_MARKDOWN_CHARS
            return cached[:MAX_MARKDOWN_CHARS], truncated

    try:
        markdown = await _fetch_via_jina(url)
    except Exception as exc:
        logger.warning("Jina Reader failed for %s (%s) — falling back to direct fetch", url, exc)
        markdown = await _fetch_direct_fallback(url)

    _scrape_cache[url] = (now, markdown)
    truncated = len(markdown) > MAX_MARKDOWN_CHARS
    return markdown[:MAX_MARKDOWN_CHARS], truncated


async def resolve_form_action(url: str) -> str:
    """BUGFIX: fetches the raw HTML directly (bypassing Jina, which strips
    the <form action=...> attribute during markdown conversion) to find the
    form's real submit target, resolved to an absolute URL. Falls back to
    the page's own URL (the previous, incorrect, always-used behavior) only
    if no form/action can be found, so submit_form() actually posts
    somewhere meaningful instead of always POSTing back to the page URL."""
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _DIRECT_FETCH_UA})
            resp.raise_for_status()
        tree = HTMLParser(resp.text)
        form = tree.css_first("form")
        action = form.attributes.get("action") if form else None
        if not action:
            return url
        return urljoin(url, action)
    except Exception as exc:
        logger.warning("resolve_form_action failed for %s (%s); falling back to page URL", url, exc)
        return url


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

    form_action_url must be the real, resolved form target — callers
    should obtain it via resolve_form_action(url) rather than passing the
    page's own URL (see BUGFIX note in this module's docstring: passing the
    page URL here used to be the only option and meant the POST almost
    never actually hit the form's real endpoint).
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
