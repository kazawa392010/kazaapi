"""
services/actions.py
--------------------
BUGFIX (see CHANGES.md #3): this is the missing "hands" for Uri.

`services/llm.route_tool_call()` decides *whether* a tool is needed and
with what arguments (a cheap classification pass). This module actually
*runs* that tool by calling straight into the existing, already-correct
service/database functions (services.tools, services.agent,
services.search, database) — no new business logic is introduced here,
this purely wires the LLM decision to the code that already existed but
was previously only reachable via a manually-authenticated REST call from
the frontend, never from natural conversation.

Every function returns a `ToolOutcome`:
  - ok           : whether the tool executed successfully
  - summary      : short plain-text description fed back to the LLM so it
                    can narrate/use the result in its reply
  - payload      : structured data sent to the client as an SSE
                    `tool_result` event (e.g. the plot image data URI) so
                    the frontend can render it directly instead of relying
                    on Uri to describe a base64 blob in words
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import database
from services import agent, search as search_service
from services import tools as tools_service

logger = logging.getLogger("kazaapi.actions")


@dataclass
class ToolOutcome:
    ok: bool
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "có", "co")


async def _do_plot(args: dict[str, Any]) -> ToolOutcome:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return ToolOutcome(False, "No expression was given to plot.")
    x_min = _as_float(args.get("x_min"), -10)
    x_max = _as_float(args.get("x_max"), 10)
    title = args.get("title")
    try:
        data_uri = tools_service.plot_expression(expression, x_min, x_max, title)
    except tools_service.PlotError as exc:
        return ToolOutcome(False, f"Could not plot '{expression}': {exc}")
    return ToolOutcome(
        True,
        f"Plotted y = {expression} for x in [{x_min}, {x_max}]. The image is attached — refer to it, don't redescribe the raw data.",
        {"type": "plot", "expression": expression, "x_min": x_min, "x_max": x_max, "data_uri": data_uri},
    )


async def _do_run_code(args: dict[str, Any]) -> ToolOutcome:
    code = str(args.get("code", ""))
    language = str(args.get("language", "python")).strip() or "python"
    stdin = str(args.get("stdin", "") or "")
    if not code.strip():
        return ToolOutcome(False, "No code was given to run.")
    try:
        result = await tools_service.run_code(code, language, None, stdin)
    except tools_service.CodeExecutionError as exc:
        return ToolOutcome(False, f"Could not execute the {language} code: {exc}")
    summary = (
        f"Ran the {language} code. stdout: {result['stdout'][:1200] or '(empty)'} "
        f"| stderr: {result['stderr'][:600] or '(empty)'} | exit_code: {result['exit_code']}"
    )
    return ToolOutcome(True, summary, {"type": "run_code", "language": language, **result})


async def _do_scrape(args: dict[str, Any]) -> ToolOutcome:
    url = str(args.get("url", "")).strip()
    if not url:
        return ToolOutcome(False, "No URL was given to read.")
    try:
        markdown, truncated = await agent.scrape_url(url)
    except Exception as exc:
        return ToolOutcome(False, f"Could not read {url}: {exc}")
    if not markdown.strip():
        return ToolOutcome(False, f"Fetched {url} but got no readable content back.")
    return ToolOutcome(
        True,
        f"Content read from {url} (truncated={truncated}):\n{markdown[:3500]}",
        {"type": "scrape", "url": url, "truncated": truncated, "markdown": markdown[:6000]},
    )


async def _do_truth_table(args: dict[str, Any]) -> ToolOutcome:
    variables_raw = str(args.get("variables", ""))
    expression = str(args.get("expression", "")).strip()
    variables = [v.strip().upper() for v in variables_raw.split(",") if v.strip()]
    if not variables or not expression:
        return ToolOutcome(False, "Need at least one variable and a boolean expression.")
    try:
        rows = tools_service.generate_truth_table(variables, expression)
    except tools_service.LogicExpressionError as exc:
        return ToolOutcome(False, f"Could not evaluate '{expression}': {exc}")
    return ToolOutcome(
        True,
        f"Truth table for '{expression}' over {variables}: {rows}",
        {"type": "truth_table", "variables": variables, "expression": expression, "rows": rows},
    )


async def _do_search_images(args: dict[str, Any]) -> ToolOutcome:
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolOutcome(False, "No image search query was given.")
    try:
        results = await search_service.web_search(f"{query} image", max_results=6)
    except Exception as exc:
        return ToolOutcome(False, f"Image search failed: {exc}")
    if not results:
        return ToolOutcome(False, f"No image results found for '{query}'.")
    listing = "; ".join(f"{r.title} ({r.url})" for r in results)
    return ToolOutcome(
        True,
        f"Image search results for '{query}': {listing}",
        {"type": "search_images", "query": query, "results": [{"url": r.url, "title": r.title} for r in results]},
    )


async def _do_flashcard_create(args: dict[str, Any]) -> ToolOutcome:
    deck = str(args.get("deck", "")).strip() or "general"
    front = str(args.get("front", "")).strip()
    back = str(args.get("back", "")).strip()
    if not front or not back:
        return ToolOutcome(False, "Need both a front and back to create a flashcard.")
    row = {
        "id": __import__("uuid").uuid4().hex,
        "deck": deck,
        "front": front,
        "back": back,
        "repetitions": 0,
        "ease_factor": 2.5,
        "interval_days": 0,
        "due_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        created = await database.create_flashcard(row)
    except database.DatabaseError as exc:
        return ToolOutcome(False, f"Could not save the flashcard: {exc}")
    card = {**row, **created}
    return ToolOutcome(True, f"Saved flashcard in deck '{deck}': {front} -> {back}",
                        {"type": "flashcard_created", "card": card})


async def _do_flashcard_list(args: dict[str, Any]) -> ToolOutcome:
    deck = args.get("deck") or None
    due_only = _as_bool(args.get("due_only", False))
    rows = await database.list_flashcards(deck=deck, due_only=due_only)
    if not rows:
        return ToolOutcome(True, "No flashcards found matching that filter.",
                            {"type": "flashcard_list", "cards": []})
    listing = "; ".join(f"[{c.get('deck')}] {c.get('front')} -> {c.get('back')}" for c in rows[:20])
    return ToolOutcome(True, f"{len(rows)} flashcard(s) found: {listing}",
                        {"type": "flashcard_list", "cards": rows})


async def _do_keep_note_add(args: dict[str, Any]) -> ToolOutcome:
    title = str(args.get("title", "")).strip()
    content = str(args.get("content", "")).strip()
    remind_at = args.get("remind_at")
    if not title:
        return ToolOutcome(False, "Need a title to save a note/reminder.")
    note = tools_service.keep_mock.add_note(title, content, remind_at)
    return ToolOutcome(True, f"Saved note '{title}'.", {"type": "keep_note", "note": note})


async def _do_web_search(args: dict[str, Any]) -> ToolOutcome:
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolOutcome(False, "No search query was given.")
    try:
        results = await search_service.web_search(query, max_results=5)
    except Exception as exc:
        return ToolOutcome(False, f"Web search failed: {exc}")
    if not results:
        return ToolOutcome(False, f"No web results found for '{query}'.")
    summary = search_service.format_results_for_prompt(results)
    return ToolOutcome(True, summary,
                        {"type": "web_search", "query": query,
                         "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]})


_DISPATCH = {
    "plot": _do_plot,
    "run_code": _do_run_code,
    "scrape": _do_scrape,
    "truth_table": _do_truth_table,
    "search_images": _do_search_images,
    "flashcard_create": _do_flashcard_create,
    "flashcard_list": _do_flashcard_list,
    "keep_note_add": _do_keep_note_add,
    "web_search": _do_web_search,
}


async def dispatch_tool(name: str, args: dict[str, Any]) -> Optional[ToolOutcome]:
    """Executes `name` with `args`. Returns None if `name` isn't a known
    tool (caller should treat that the same as 'no tool'). Never raises —
    every branch above already catches its own domain errors, and this is
    the last line of defense so a single bad tool call can never crash the
    whole chat turn (SECTION 8: 'graceful fallback ... log errors but
    never crash')."""
    fn = _DISPATCH.get(name)
    if fn is None:
        logger.warning("dispatch_tool: unknown tool %r requested", name)
        return None
    try:
        return await fn(args or {})
    except Exception as exc:  # belt-and-suspenders: a tool must never 500 the chat turn
        logger.exception("dispatch_tool: tool %r raised unexpectedly", name)
        return ToolOutcome(False, f"Tool '{name}' failed unexpectedly: {exc}")
