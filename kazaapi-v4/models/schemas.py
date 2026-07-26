"""
models/schemas.py
------------------
Central Pydantic models for requests/responses. Keeping these in one module
(instead of scattered inline dicts) means FastAPI can auto-generate accurate
OpenAPI docs and we get free request validation, which saves us from writing
manual `if "x" not in body` checks that would otherwise burn CPU cycles on
Render's throttled free tier.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """RBAC roles. 'kaza' is the owner/admin, 'stranger' is everyone else."""
    kaza = "kaza"
    stranger = "stranger"


class EmotionState(str, Enum):
    amused = "amused"
    annoyed = "annoyed"
    serious = "serious"
    indifferent = "indifferent"


class Mode(str, Enum):
    casual = "casual"
    task = "task"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session/thread id")
    message: str = Field(..., min_length=1, max_length=8000)
    role: Role = Role.stranger
    mode: Optional[Mode] = None  # if None, backend infers casual vs task


class EmotionVector(BaseModel):
    """Probability distribution over Uri's emotional states. Always sums to 1."""
    amused: float = 0.25
    annoyed: float = 0.25
    serious: float = 0.25
    indifferent: float = 0.25

    def normalized(self) -> "EmotionVector":
        total = self.amused + self.annoyed + self.serious + self.indifferent
        if total <= 0:
            return EmotionVector()
        return EmotionVector(
            amused=self.amused / total,
            annoyed=self.annoyed / total,
            serious=self.serious / total,
            indifferent=self.indifferent / total,
        )


class ChatHistoryItem(BaseModel):
    id: Optional[str] = None
    session_id: str
    sender_role: str  # "user" | "assistant" | "system"
    content: str
    created_at: Optional[datetime] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryItem]
    long_term_summary: Optional[str] = None


class ChatResetResponse(BaseModel):
    session_id: str
    cleared: bool


# ---------------------------------------------------------------------------
# Tools: plotting
# ---------------------------------------------------------------------------

class PlotRequest(BaseModel):
    expression: str = Field(..., description="e.g. 'sin(x) + x**2'")
    x_min: float = -10
    x_max: float = 10
    title: Optional[str] = None


class PlotResponse(BaseModel):
    data_uri: str  # base64 PNG, e.g. "data:image/png;base64,...."
    expression: str


# ---------------------------------------------------------------------------
# Tools: code execution (Piston)
# ---------------------------------------------------------------------------

class RunCodeRequest(BaseModel):
    code: str
    language: str = Field(..., description="e.g. python, javascript, cpp, java")
    version: Optional[str] = Field(None, description="Piston language version, '*' if omitted")
    stdin: Optional[str] = ""


class RunCodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: Optional[int] = None
    language: str
    version: Optional[str] = None


# ---------------------------------------------------------------------------
# Tools: scraping / form submission
# ---------------------------------------------------------------------------

class ScrapeRequest(BaseModel):
    url: str


class ScrapeResponse(BaseModel):
    url: str
    markdown: str
    truncated: bool = False


class FormField(BaseModel):
    name: str
    label: Optional[str] = None
    field_type: str = "text"
    required: bool = False
    value: Optional[str] = None


class SubmitFormRequest(BaseModel):
    url: str
    known_values: dict[str, str] = Field(default_factory=dict)


class SubmitFormResponse(BaseModel):
    status: str  # "needs_input" | "submitted" | "error"
    missing_fields: list[FormField] = Field(default_factory=list)
    summary: Optional[str] = None
    raw_excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# Tools: image search
# ---------------------------------------------------------------------------

class ImageSearchResult(BaseModel):
    url: str
    thumbnail: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None


class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageSearchResult]


# ---------------------------------------------------------------------------
# Tools: flashcards (SM-2 spaced repetition)
# ---------------------------------------------------------------------------

class FlashcardCreate(BaseModel):
    deck: str
    front: str
    back: str


class FlashcardUpdate(BaseModel):
    front: Optional[str] = None
    back: Optional[str] = None
    deck: Optional[str] = None


class FlashcardReviewRequest(BaseModel):
    card_id: str
    quality: int = Field(..., ge=0, le=5, description="SM-2 recall quality, 0=blackout, 5=perfect")


class Flashcard(BaseModel):
    id: str
    owner_role: str = "kaza"
    deck: str
    front: str
    back: str
    repetitions: int = 0
    ease_factor: float = 2.5
    interval_days: int = 0
    due_at: datetime
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Config / preferences
# ---------------------------------------------------------------------------

class PersonalitySliders(BaseModel):
    sassiness: float = Field(0.7, ge=0, le=1)
    formality_in_task_mode: float = Field(0.8, ge=0, le=1)
    proactivity: float = Field(0.3, ge=0, le=1)


class UserPreferences(BaseModel):
    personality: PersonalitySliders = Field(default_factory=PersonalitySliders)
    tool_toggles: dict[str, bool] = Field(default_factory=dict)
    # Masked on GET; only accepted (and encrypted) on PUT.
    api_keys: dict[str, str] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    personality: PersonalitySliders
    tool_toggles: dict[str, bool]
    api_keys_configured: dict[str, bool]  # which keys are set, values never returned


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
