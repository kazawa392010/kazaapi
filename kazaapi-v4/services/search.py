"""
services/search.py
-------------------
Free search integration for the "dual-speed router" (SECTION 2): before
answering fact-checking / lookup-style queries, we hit a free search
endpoint and feed the top results to the model as context.

BUGFIX (see CHANGES.md #1 — "không đọc được web"): this used to rely
*solely* on DuckDuckGo's HTML endpoint. In practice that endpoint routinely
serves an anti-bot/"unusual traffic" challenge page to requests coming from
data-center IPs (exactly what Render's free tier uses), which the old code
didn't detect — it just parsed the challenge page, found zero `div.result`
nodes, and silently returned `[]` every time. From the outside that looks
exactly like "Uri doesn't know how to read the web", because every
search-grounded question and every scrape-adjacent feature (self-learning,
image search) quietly got nothing back.

Fix: try Jina's free search endpoint (https://s.jina.ai/<query>, no key
required, same family of service as the Jina Reader already used in
agent.py) FIRST, since it's markdown/JSON based and far less prone to
anti-bot walls; fall back to the DuckDuckGo HTML scraper second. Both
paths are still wrapped so a total failure returns [] rather than raising
— callers should keep treating "no results" as "answer from the model's
own knowledge" rather than a hard error.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger("kazaapi.search")

SEARCH_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
JINA_SEARCH_BASE = "https://s.jina.ai/"

# A real browser UA + Accept-Language noticeably reduces how often DDG's
# HTML endpoint serves an anti-bot challenge instead of real results.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

_DDG_BLOCK_MARKERS = ("anomaly", "unusual traffic", "captcha", "detected unusual")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean_ddg_redirect(href: str) -> str:
    """DDG's HTML endpoint wraps result links in a redirect like
    //duckduckgo.com/l/?uddg=<url-encoded-target>&rut=..."""
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    return href


async def _search_jina(query: str, max_results: int) -> list[SearchResult]:
    """Free, keyless (or optionally keyed via JINA_KEY for higher rate
    limits) search via r.jina.ai's sister endpoint s.jina.ai. Returns
    clean JSON, sidestepping HTML-scraping fragility entirely."""
    key = os.environ.get("JINA_KEY")
    headers = {**_BROWSER_HEADERS, "Accept": "application/json", "X-Respond-With": "no-content"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    url = JINA_SEARCH_BASE + quote(query, safe="")
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
    data = resp.json()
    items = data.get("data") or []

    results: list[SearchResult] = []
    for item in items:
        u = item.get("url")
        t = item.get("title")
        snippet = item.get("description") or (item.get("content") or "")[:220]
        if u and t:
            results.append(SearchResult(title=t, url=u, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


async def _search_ddg(query: str, max_results: int) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.post(DDG_HTML_URL, data={"q": query}, headers=_BROWSER_HEADERS)
        resp.raise_for_status()

    lowered_body = resp.text.lower()
    if any(marker in lowered_body for marker in _DDG_BLOCK_MARKERS):
        raise RuntimeError("DuckDuckGo served an anti-bot challenge page instead of results")

    tree = HTMLParser(resp.text)
    results: list[SearchResult] = []
    for node in tree.css("div.result"):
        title_node = node.css_first("a.result__a")
        snippet_node = node.css_first("a.result__snippet, div.result__snippet")
        if not title_node:
            continue
        url = _clean_ddg_redirect(title_node.attributes.get("href", ""))
        title = title_node.text(strip=True)
        snippet = snippet_node.text(strip=True) if snippet_node else ""
        if url and title:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Best-effort free web search with graceful multi-backend fallback
    (SECTION 8: "graceful fallback for every external API"). Tries Jina
    Search first, then DuckDuckGo's HTML endpoint. Returns [] only if both
    fail — callers should treat that as "answer from the model's own
    knowledge", not as a hard error."""
    try:
        results = await _search_jina(query, max_results)
        if results:
            return results
        logger.info("Jina search returned no results for %r, trying DuckDuckGo", query)
    except Exception as exc:
        logger.warning("Jina search failed for %r: %s — falling back to DuckDuckGo", query, exc)

    try:
        return await _search_ddg(query, max_results)
    except Exception as exc:
        logger.warning("web_search (DuckDuckGo fallback) failed for %r: %s", query, exc)
        return []


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """Turns search results into a compact context block for the LLM prompt."""
    if not results:
        return ""
    lines = ["Web search results (use these to ground your answer, cite sources briefly):"]
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.title} — {r.snippet} ({r.url})")
    return "\n".join(lines)


def looks_like_factual_query(message: str) -> bool:
    """Cheap heuristic fallback: does this message likely need current/
    factual grounding rather than pure conversation? Used only when the
    tool-routing classifier (services/llm.route_tool_call) didn't already
    pick the web_search tool, as a safety net. BUGFIX: expanded Vietnamese
    coverage — the original list only had 3 Vietnamese phrases, so most
    Vietnamese factual questions never triggered search grounding at all."""
    signals = (
        # English
        "who is", "what is", "when did", "when is", "latest", "current",
        "news", "price of", "how many", "score", "result of",
        # Vietnamese
        "định nghĩa", "là gì", "hôm nay", "ai là", "cái gì", "tại sao",
        "như thế nào", "bao nhiêu", "khi nào", "ở đâu", "giá bao nhiêu",
        "tin tức", "mới nhất", "hiện tại", "bây giờ",
    )
    lowered = message.lower()
    return any(s in lowered for s in signals)
