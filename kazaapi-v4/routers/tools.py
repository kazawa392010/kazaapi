"""
routers/tools.py
-----------------
All SECTION 4 backend tools, exposed as REST endpoints. Every tool runs
entirely server-side; the frontend only renders whatever JSON/base64 comes
back (SECTION 4's "no heavy simulation on server" for circuits, and
"no file storage needed" for plots, are both honored here).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import database
from utils.auth import require_kaza
from models.schemas import (
    Flashcard,
    FlashcardCreate,
    FlashcardReviewRequest,
    FlashcardUpdate,
    FormField,
    ImageSearchResponse,
    ImageSearchResult,
    PlotRequest,
    PlotResponse,
    RunCodeRequest,
    RunCodeResponse,
    ScrapeRequest,
    ScrapeResponse,
    SubmitFormRequest,
    SubmitFormResponse,
)
from services import agent, tools as tools_service
from services.search import web_search

router = APIRouter(
    prefix="/api/v1/tools",
    tags=["tools"],
    # SECTION 1: strangers get "no tool execution, no memory write" — every
    # route in this router requires a valid X-Kaza-Token header.
    dependencies=[Depends(require_kaza)],
)


# ---------------------------------------------------------------------------
# Mathematics: plotting
# ---------------------------------------------------------------------------

@router.post("/plot", response_model=PlotResponse)
async def plot(req: PlotRequest):
    try:
        data_uri = tools_service.plot_expression(req.expression, req.x_min, req.x_max, req.title)
    except tools_service.PlotError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PlotResponse(data_uri=data_uri, expression=req.expression)


# ---------------------------------------------------------------------------
# Computer Science: code execution
# ---------------------------------------------------------------------------

@router.post("/run_code", response_model=RunCodeResponse)
async def run_code(req: RunCodeRequest):
    try:
        result = await tools_service.run_code(req.code, req.language, req.version, req.stdin or "")
    except tools_service.CodeExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RunCodeResponse(**result)


@router.get("/run_code/runtimes")
async def run_code_runtimes():
    try:
        return await tools_service.list_piston_runtimes()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch Piston runtimes: {exc}") from exc


# ---------------------------------------------------------------------------
# Technology: logic / truth table
# ---------------------------------------------------------------------------

@router.get("/logic/truth_table")
async def truth_table(variables: str = Query(..., description="comma-separated, e.g. A,B,C"),
                       expression: str = Query(..., description="e.g. 'A AND B OR NOT C'")):
    var_list = [v.strip().upper() for v in variables.split(",") if v.strip()]
    try:
        rows = tools_service.generate_truth_table(var_list, expression)
    except tools_service.LogicExpressionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"variables": var_list, "expression": expression, "rows": rows,
            "circuit_builder_url": tools_service.CIRCUITVERSE_NEW_PROJECT_URL}


# ---------------------------------------------------------------------------
# Web scraping & form submission
# ---------------------------------------------------------------------------

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape(req: ScrapeRequest):
    try:
        markdown, truncated = await agent.scrape_url(req.url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Scrape failed: {exc}") from exc
    return ScrapeResponse(url=req.url, markdown=markdown, truncated=truncated)


@router.post("/submit_form", response_model=SubmitFormResponse)
async def submit_form(req: SubmitFormRequest):
    try:
        markdown, _ = await agent.scrape_url(req.url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load form page: {exc}") from exc

    fields = await agent.extract_form_fields(markdown)
    if not fields:
        return SubmitFormResponse(status="error", summary="No submittable form found on that page.")

    missing = agent.diff_missing_fields(fields, req.known_values)
    if missing:
        return SubmitFormResponse(status="needs_input", missing_fields=missing)

    try:
        result = await agent.submit_form(req.url, req.url, req.known_values)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Form submission failed: {exc}") from exc

    return SubmitFormResponse(
        status=result["status"],
        summary=result.get("summary"),
        raw_excerpt=result.get("raw_excerpt"),
    )


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------

@router.get("/search_images", response_model=ImageSearchResponse)
async def search_images(query: str = Query(...), max_results: int = Query(6, ge=1, le=20)):
    """Best-effort free image search. There's no official free image-search
    JSON API, so we reuse the DuckDuckGo HTML backend's regular web results
    filtered to obviously image-hosting domains/pages as a pragmatic
    approximation, and clearly label results as best-effort. For production
    use, plug in a real (still-free-tier) key such as Bing Image Search's
    trial or SerpAPI's free monthly quota via an env var and swap the
    implementation below without touching the router contract."""
    try:
        results = await web_search(f"{query} image", max_results=max_results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image search failed: {exc}") from exc

    return ImageSearchResponse(
        query=query,
        results=[
            ImageSearchResult(url=r.url, source=r.url, title=r.title)
            for r in results
        ],
    )


# ---------------------------------------------------------------------------
# Flashcards (CRUD + SM-2 review)
# ---------------------------------------------------------------------------

@router.post("/flashcard", response_model=Flashcard)
async def create_flashcard(card: FlashcardCreate):
    row = {
        "id": str(uuid.uuid4()),
        "deck": card.deck,
        "front": card.front,
        "back": card.back,
        "repetitions": 0,
        "ease_factor": 2.5,
        "interval_days": 0,
        "due_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        created = await database.create_flashcard(row)
    except database.DatabaseError as exc:
        raise HTTPException(status_code=502, detail=f"Could not save flashcard: {exc}") from exc
    return Flashcard(**{**row, **created})


@router.get("/flashcard", response_model=list[Flashcard])
async def list_flashcards(deck: Optional[str] = None, due_only: bool = False):
    rows = await database.list_flashcards(deck=deck, due_only=due_only)
    return [Flashcard(**row) for row in rows]


@router.patch("/flashcard/{card_id}", response_model=Flashcard)
async def update_flashcard(card_id: str, patch: FlashcardUpdate):
    existing = await database.get_flashcard(card_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    try:
        updated = await database.update_flashcard(card_id, patch.model_dump(exclude_none=True))
    except database.DatabaseError as exc:
        raise HTTPException(status_code=502, detail=f"Could not update flashcard: {exc}") from exc
    return Flashcard(**{**existing, **updated})


@router.delete("/flashcard/{card_id}")
async def delete_flashcard(card_id: str):
    existing = await database.get_flashcard(card_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    try:
        await database.delete_flashcard(card_id)
    except database.DatabaseError as exc:
        raise HTTPException(status_code=502, detail=f"Could not delete flashcard: {exc}") from exc
    return {"deleted": True, "id": card_id}


@router.post("/flashcard/review", response_model=Flashcard)
async def review_flashcard(req: FlashcardReviewRequest):
    card = await database.get_flashcard(req.card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    sm2 = tools_service.sm2_review(
        quality=req.quality,
        repetitions=card.get("repetitions", 0),
        ease_factor=card.get("ease_factor", 2.5),
        interval_days=card.get("interval_days", 0),
    )
    try:
        updated = await database.update_flashcard(req.card_id, sm2)
    except database.DatabaseError as exc:
        raise HTTPException(status_code=502, detail=f"Could not save review: {exc}") from exc
    return Flashcard(**{**card, **updated})


# ---------------------------------------------------------------------------
# Google Keep (mock) — role gating enforced by caller / API gateway layer
# ---------------------------------------------------------------------------

@router.post("/keep/notes")
async def add_keep_note(title: str, content: str, remind_at: Optional[str] = None):
    return tools_service.keep_mock.add_note(title, content, remind_at)


@router.get("/keep/notes")
async def list_keep_notes():
    return tools_service.keep_mock.list_notes()


@router.delete("/keep/notes/{note_id}")
async def delete_keep_note(note_id: str):
    ok = tools_service.keep_mock.delete_note(note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"deleted": True, "id": note_id}
