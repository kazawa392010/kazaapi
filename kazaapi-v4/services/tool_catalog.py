"""
services/tool_catalog.py
-------------------------
BUGFIX (see CHANGES.md item #3 — "tính năng yêu cầu Uri làm nhưng không
chạy"): the original codebase described a rich toolkit (plot, run code,
scrape, flashcards, ...) in Uri's system prompt, but the chat pipeline
(routers/chat.py -> services/llm.py) never actually gave the LLM any way to
*invoke* those tools. The model could only ever talk *about* the tools in
plain text, so any request like "vẽ đồ thị y = x^2 giúp tui" or "chạy đoạn
code này" always ended in Uri saying it couldn't do it — because it
genuinely had no mechanism to do it.

This module is the single source of truth for "what tools exist, and what
arguments do they take" — used by:
  - services/llm.py (to build the tool-routing classification prompt)
  - services/actions.py (to actually execute a chosen tool)

Kept dependency-free (no imports of agent/tools/database/llm) specifically
so it can be imported from both llm.py and actions.py without creating an
import cycle (agent.py already imports llm.py).
"""

from __future__ import annotations

from typing import Any

# Each entry:
#   name          -- must match the dispatcher key in services/actions.py
#   description    -- shown to the routing-classifier model
#   parameters     -- {param_name: "human description of expected value"}
#   required       -- list of parameter names that must be present
#   kaza_only      -- True if SECTION 1 RBAC restricts this to role == 'kaza'
#                      (i.e. anything that is a real "tool execution", per
#                      "stranger: ... no tool execution, no memory write")
TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "plot",
        "description": "Plot a single-variable math function y = f(x) over a range and return a graph image.",
        "parameters": {
            "expression": "the function of x to plot, e.g. 'sin(x) + x**2'",
            "x_min": "optional lower bound of x (number, default -10)",
            "x_max": "optional upper bound of x (number, default 10)",
            "title": "optional chart title",
        },
        "required": ["expression"],
        "kaza_only": True,
    },
    {
        "name": "run_code",
        "description": "Execute a snippet of code (Python, JavaScript, C++, Java, etc.) in a sandbox and return stdout/stderr.",
        "parameters": {
            "code": "the full source code to run",
            "language": "the language name, e.g. 'python', 'javascript', 'cpp', 'java'",
            "stdin": "optional stdin input to feed the program",
        },
        "required": ["code", "language"],
        "kaza_only": True,
    },
    {
        "name": "scrape",
        "description": "Fetch and read the text content of a specific web page URL (use when the user gives/asks about a link).",
        "parameters": {"url": "the absolute URL to read"},
        "required": ["url"],
        "kaza_only": True,
    },
    {
        "name": "truth_table",
        "description": "Generate a boolean truth table for a logic expression built from AND/OR/XOR/NAND/NOR/NOT.",
        "parameters": {
            "variables": "comma-separated variable names, e.g. 'A,B,C'",
            "expression": "the boolean expression, e.g. 'A AND B OR NOT C'",
        },
        "required": ["variables", "expression"],
        "kaza_only": True,
    },
    {
        "name": "search_images",
        "description": "Search the web for images matching a query.",
        "parameters": {"query": "what to search images for"},
        "required": ["query"],
        "kaza_only": True,
    },
    {
        "name": "flashcard_create",
        "description": "Create and save a new spaced-repetition flashcard.",
        "parameters": {
            "deck": "name of the deck/collection this card belongs to",
            "front": "the question/prompt side of the card",
            "back": "the answer side of the card",
        },
        "required": ["deck", "front", "back"],
        "kaza_only": True,
    },
    {
        "name": "flashcard_list",
        "description": "List saved flashcards, optionally filtered to a deck or only cards due for review right now.",
        "parameters": {
            "deck": "optional deck name to filter by",
            "due_only": "optional true/false — only return cards due now",
        },
        "required": [],
        "kaza_only": True,
    },
    {
        "name": "keep_note_add",
        "description": "Save a reminder/note (mocked Google Keep).",
        "parameters": {
            "title": "short note title",
            "content": "note body/details",
            "remind_at": "optional ISO-8601 datetime to remind at",
        },
        "required": ["title", "content"],
        "kaza_only": True,
    },
    {
        "name": "web_search",
        "description": "Search the live web for current/factual information you might not already know (news, prices, definitions, 'who/what/when' questions, etc.).",
        "parameters": {"query": "the search query"},
        "required": ["query"],
        "kaza_only": False,
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_CATALOG}


def available_tools(is_kaza: bool) -> list[dict[str, Any]]:
    """Tools a given role is allowed to trigger."""
    return [t for t in TOOL_CATALOG if is_kaza or not t["kaza_only"]]
