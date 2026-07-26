"""
services/tools.py
------------------
Subject-specific server-side tools (SECTION 4.C):
  - Mathematics: SymPy + Matplotlib graph plotting -> base64 PNG data URI
  - Computer Science: Piston API code execution
  - Technology: lightweight truth-table generator for logic circuits
  - Flashcards: SM-2 spaced repetition scheduling
  - Google Keep: mocked reminder/notes service (role == 'kaza' only,
    enforced by the router, not here)

Matplotlib is used with the non-interactive 'Agg' backend so it never tries
to open a display (would crash/hang on a headless Render worker) and never
writes to disk (we return the PNG as bytes straight from an in-memory
buffer, per SECTION 4.C's "no file storage needed").
"""

from __future__ import annotations

import base64
import io
import itertools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import matplotlib
matplotlib.use("Agg")  # headless, no GUI backend, no disk writes
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)

logger = logging.getLogger("kazaapi.tools")

PISTON_URL = "https://emkc.org/api/v2/piston/execute"
PISTON_RUNTIMES_URL = "https://emkc.org/api/v2/piston/runtimes"
PISTON_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_ALLOWED_SYMPY_NAMES = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "pi", "E", "Abs", "factorial",
    )
}


# ---------------------------------------------------------------------------
# Mathematics: plotting
# ---------------------------------------------------------------------------

class PlotError(Exception):
    pass


def plot_expression(expression: str, x_min: float, x_max: float, title: str | None) -> str:
    """Parses `expression` with SymPy (safely, no eval()) and renders a PNG
    with Matplotlib, returning a data: URI string."""
    x = sp.symbols("x")
    local_dict = {**_ALLOWED_SYMPY_NAMES, "x": x}
    try:
        parsed = parse_expr(expression, local_dict=local_dict, transformations=_TRANSFORMS)
    except Exception as exc:
        raise PlotError(f"Could not parse expression '{expression}': {exc}") from exc

    func = sp.lambdify(x, parsed, modules=["numpy"])

    xs = np.linspace(x_min, x_max, 800)
    try:
        with np.errstate(all="ignore"):
            ys = func(xs)
            ys = np.asarray(ys, dtype=float)
            ys = np.where(np.isfinite(ys), ys, np.nan)
    except Exception as exc:
        raise PlotError(f"Could not evaluate expression over range: {exc}") from exc

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.plot(xs, ys, linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.grid(True, linewidth=0.3)
    ax.set_title(title or f"y = {expression}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)  # critical: free the figure, otherwise RAM leaks across requests
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Computer Science: code execution via Piston
# ---------------------------------------------------------------------------

class CodeExecutionError(Exception):
    pass


async def run_code(code: str, language: str, version: str | None, stdin: str = "") -> dict[str, Any]:
    payload = {
        "language": language,
        "version": version or "*",
        "files": [{"content": code}],
        "stdin": stdin or "",
    }
    async with httpx.AsyncClient(timeout=PISTON_TIMEOUT) as client:
        try:
            resp = await client.post(PISTON_URL, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CodeExecutionError(
                f"Piston returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            raise CodeExecutionError(f"Could not reach Piston API: {exc}") from exc

    data = resp.json()
    run = data.get("run", {})
    return {
        "stdout": run.get("stdout", ""),
        "stderr": run.get("stderr", ""),
        "exit_code": run.get("code"),
        "language": data.get("language", language),
        "version": data.get("version", version),
    }


async def list_piston_runtimes() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=PISTON_TIMEOUT) as client:
        resp = await client.get(PISTON_RUNTIMES_URL)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Technology: truth-table generator (lightweight logic sim, SECTION 4.C)
# ---------------------------------------------------------------------------

class LogicExpressionError(Exception):
    pass

_LOGIC_OPS = {
    "AND": lambda a, b: a and b,
    "OR": lambda a, b: a or b,
    "XOR": lambda a, b: a != b,
    "NAND": lambda a, b: not (a and b),
    "NOR": lambda a, b: not (a or b),
    "NOT": lambda a: not a,
}


def generate_truth_table(variables: list[str], expression: str) -> list[dict[str, Any]]:
    """expression example: 'A AND B', 'NOT A OR B', 'A XOR B AND C'
    Evaluated left-to-right token by token (simple, intentionally not a full
    parser — good enough for intro logic-gate homework, and avoids pulling
    in a heavy grammar/parsing dependency on a memory-constrained server)."""
    if not variables:
        raise LogicExpressionError("At least one variable is required.")

    tokens = expression.upper().split()
    rows = []
    for combo in itertools.product([False, True], repeat=len(variables)):
        env = dict(zip(variables, combo))
        try:
            result = _eval_logic_tokens(tokens, env)
        except Exception as exc:
            raise LogicExpressionError(f"Could not evaluate '{expression}': {exc}") from exc
        row = {var: val for var, val in zip(variables, combo)}
        row["result"] = result
        rows.append(row)
    return rows


def _eval_logic_tokens(tokens: list[str], env: dict[str, bool]) -> bool:
    """Evaluates a space-separated boolean expression with standard
    precedence: NOT binds tightest (unary prefix), then AND/NAND, then
    OR/XOR/NOR left-to-right. No parentheses support — for anything more
    complex than a couple of terms, ask the user to split it into multiple
    named expressions rather than trying to parse nested parens here.
    """
    # Pass 1: resolve operands (applying any NOT prefixes) and collect the
    # binary operators between them, e.g. "A OR B AND C" -> values=[A,B,C], ops=[OR,AND]
    values: list[bool] = []
    ops: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "NOT":
            i += 1
            if i >= len(tokens) or tokens[i] not in env:
                raise LogicExpressionError("NOT must be followed by a known variable.")
            values.append(not env[tokens[i]])
        elif tok in env:
            values.append(env[tok])
        elif tok in _LOGIC_OPS:  # AND/OR/XOR/NAND/NOR (NOT already handled above)
            ops.append(tok)
        else:
            raise LogicExpressionError(f"Unknown token '{tok}'")
        i += 1

    if len(values) != len(ops) + 1:
        raise LogicExpressionError("Malformed expression (check operand/operator count).")

    # Pass 2: resolve higher-precedence AND/NAND first.
    reduced_values = [values[0]]
    reduced_ops: list[str] = []
    for op, val in zip(ops, values[1:]):
        if op in ("AND", "NAND"):
            left = reduced_values.pop()
            reduced_values.append(_LOGIC_OPS[op](left, val))
        else:
            reduced_values.append(val)
            reduced_ops.append(op)

    # Pass 3: resolve remaining OR/XOR/NOR left-to-right.
    result = reduced_values[0]
    for op, val in zip(reduced_ops, reduced_values[1:]):
        result = _LOGIC_OPS[op](result, val)
    return bool(result)


CIRCUITVERSE_NEW_PROJECT_URL = "https://circuitverse.org/simulator/new"


# ---------------------------------------------------------------------------
# Flashcards: SM-2 spaced repetition
# ---------------------------------------------------------------------------

def sm2_review(quality: int, repetitions: int, ease_factor: float, interval_days: int) -> dict[str, Any]:
    """Classic SuperMemo-2 algorithm.
    quality: 0-5 recall rating from the user for this review.
    """
    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)
        repetitions += 1

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(1.3, ease_factor)

    due_at = datetime.now(timezone.utc) + timedelta(days=interval_days)
    return {
        "repetitions": repetitions,
        "ease_factor": round(ease_factor, 3),
        "interval_days": interval_days,
        "due_at": due_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Google Keep — mock service (SECTION 4.C, role == 'kaza' only)
# ---------------------------------------------------------------------------

class KeepMock:
    """In-memory mock of a notes/reminders service. There is no free public
    Google Keep API, so this satisfies the spec's 'Mock / OAuth' fallback.
    Swap for a real integration (e.g. Google Tasks API, which *does* have a
    free OAuth-based API) by replacing this class's methods."""

    def __init__(self) -> None:
        self._notes: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def add_note(self, title: str, content: str, remind_at: str | None = None) -> dict[str, Any]:
        self._counter += 1
        note_id = f"mock-{self._counter}"
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "remind_at": remind_at,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._notes[note_id] = note
        return note

    def list_notes(self) -> list[dict[str, Any]]:
        return list(self._notes.values())

    def delete_note(self, note_id: str) -> bool:
        return self._notes.pop(note_id, None) is not None


keep_mock = KeepMock()  # process-lifetime singleton; fine since it's a mock
