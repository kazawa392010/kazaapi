"""
services/search.py
-------------------
Free search integration for the "dual-speed router" (SECTION 2): before
answering fact-checking / lookup-style queries, we hit a free search
endpoint and feed the top results to the model as context.

There is no official free DuckDuckGo JSON search API for general web
results, so we use DuckDuckGo's HTML endpoint (html.duckduckgo.com), which
has no API key and is commonly used for this purpose. It is best-effort:
DuckDuckGo can change markup or rate-limit without notice, so every call is
wrapped and returns an empty list on failure rather than raising — the
caller (services/llm.py) should treat "no results" as "answer from the
model's own knowledge" rather than a hard error.

If you have a Jina Search key (also free), you can swap this out for
https://s.jina.ai/<query> which returns clean markdown results and tends to
be more stable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger("kazaapi.search")

SEARCH_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
DDG_HTML_URL = "https://html.duckduckgo.com/html/"


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


async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Best-effort free web search. Returns [] on any failure."""
    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.post(
                DDG_HTML_URL,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; KazaAPI/4.0)"},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return []

    try:
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
    except Exception as exc:
        logger.warning("web_search parse failed for %r: %s", query, exc)
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
    """Cheap heuristic: does this message likely need current/factual
    grounding rather than pure conversation or code help?"""
    signals = (
        "who is", "what is", "when did", "when is", "latest", "current",
        "news", "price of", "how many", "статистика", "score", "result of",
        "định nghĩa", "là gì", "hôm nay",
    )
    lowered = message.lower()
    return any(s in lowered for s in signals)
