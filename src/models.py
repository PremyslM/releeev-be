"""
Pydantic models — shared between API layer, extraction, and linking.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class CaptureType(str, Enum):
    screenshot = "screenshot"
    photo = "photo"
    document = "document"
    unknown = "unknown"


class LinkType(str, Enum):
    topic_overlap = "TOPIC_OVERLAP"
    entity_match = "ENTITY_MATCH"
    temporal_proximity = "TEMPORAL_PROXIMITY"
    embedding_similarity = "EMBEDDING_SIMILARITY"


# ── Extracted attributes ──────────────────────────────────────────────────────

class ExtractedAttributes(BaseModel):
    """Raw output from Claude Vision extraction."""
    title: str = ""
    summary: str = ""
    raw_text: str = ""
    capture_type: CaptureType = CaptureType.unknown
    topics: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    language: str = "en"
    sentiment: str = "neutral"          # positive / negative / neutral
    confidence: float = 1.0             # extraction confidence 0-1
    extra: dict[str, Any] = Field(default_factory=dict)


# ── Graph node models ──────────────────────────────────────────────────────────

class CaptureNode(BaseModel):
    """Represents a Capture node stored in Neo4j."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    image_hash: str = ""                # SHA-256 of raw image bytes
    attributes: ExtractedAttributes = Field(default_factory=ExtractedAttributes)
    embedding: list[float] = Field(default_factory=list)  # 1536-dim vector


class TopicNode(BaseModel):
    name: str
    capture_count: int = 0


class PersonNode(BaseModel):
    name: str
    capture_count: int = 0


class PlaceNode(BaseModel):
    name: str
    capture_count: int = 0


# ── Link model ─────────────────────────────────────────────────────────────────

class CaptureLink(BaseModel):
    source_id: str
    target_id: str
    link_type: LinkType
    strength: float = Field(ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)
    # Denormalized for display — filled by GET /captures/{id}
    other_title: str = ""
    other_summary: str = ""
    shared_topics: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # why these are linked


# ── API request / response models ─────────────────────────────────────────────

class AnalyzeCaptureRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded image (JPEG / PNG / WebP)",
    )
    image_mime: str = Field(
        default="image/jpeg",
        pattern=r"^image/(jpeg|png|webp|gif)$",
    )
    # Optional capture location (from photo EXIF / device GPS, reverse-geocoded
    # on the client). Turns a photo into a place-anchored memory.
    lat: float | None = None
    lon: float | None = None
    place: str | None = None        # e.g. "Kavárna 14g"
    locality: str | None = None     # e.g. "Mariánské Lázně"
    captured_at: str | None = None  # real capture time (ISO), distinct from upload


class AnalyzeUrlRequest(BaseModel):
    url: str = Field(..., description="Webová adresa ke zpracování")


class LinkedCapture(BaseModel):
    capture_id: str
    link_type: LinkType
    strength: float
    title: str = ""
    summary: str = ""
    reasons: list[str] = Field(default_factory=list)  # why these are linked


class AnalyzeCaptureResponse(BaseModel):
    capture_id: str
    attributes: ExtractedAttributes
    embedding_dim: int
    linked_to: list[LinkedCapture] = Field(default_factory=list)
    processing_ms: int = 0
    subject_name: str | None = None  # set if auto-grouped into a Subject branch


class SubjectSibling(BaseModel):
    id: str
    title: str = ""
    source: str = ""


class SubjectInfo(BaseModel):
    """The real-world thing this capture depicts, seen across several sources."""
    subject_id: str
    name: str = ""
    kind: str = ""
    siblings: list[SubjectSibling] = Field(default_factory=list)


class LocationInfo(BaseModel):
    """Where a capture was made — the anchor for a place-based memory."""
    lat: float
    lon: float
    place: str = ""
    locality: str = ""
    captured_at: str | None = None


class CaptureDetailResponse(BaseModel):
    capture: CaptureNode
    connections: list[CaptureLink] = Field(default_factory=list)
    subject: SubjectInfo | None = None
    location: LocationInfo | None = None


class SearchResult(BaseModel):
    capture_id: str
    title: str
    summary: str
    score: float
    created_at: datetime
    category: str = ""        # domain (for the coloured dot / label)
    capture_type: str = ""    # photo | document | note … (for the row icon)


class AskSource(BaseModel):
    """A capture cited in a RAG answer."""
    n: int
    capture_id: str
    title: str = ""
    category: str = ""
    capture_type: str = ""
    created_at: datetime | None = None


class AskResponse(BaseModel):
    """Answer synthesized strictly from the user's own captures, with citations."""
    question: str
    answer: str
    sources: list[AskSource] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    # Auto-derived refinement chips: {facet_type: {value: count}} over the
    # relevant matches. The UI renders these as one-click filters — the user
    # never configures anything.
    facets: dict[str, dict[str, int]] = Field(default_factory=dict)


# ── Entity navigation (browsable hubs) ──────────────────────────────────────────

class EntityRef(BaseModel):
    name: str
    kind: str          # topic | person | place | subject
    count: int = 0     # how many captures reference it


class EntityListResponse(BaseModel):
    """Landmarks for browsing — the most-referenced entities of each type."""
    topics: list[EntityRef] = Field(default_factory=list)
    people: list[EntityRef] = Field(default_factory=list)
    places: list[EntityRef] = Field(default_factory=list)
    subjects: list[EntityRef] = Field(default_factory=list)


class BugReportRequest(BaseModel):
    """A user-submitted bug: free text + an optional screenshot (base64 JPEG)."""
    title: str = ""
    description: str = ""
    image_base64: str | None = None
    app_version: str = ""
    device: str = ""


class BugReportResponse(BaseModel):
    url: str          # GitHub issue html_url
    number: int


class Theme(BaseModel):
    """A high-level theme distilled from many fine-grained topics — the handful
    of labels that capture what the whole brain is about."""
    name: str
    count: int = 0                                   # captures this theme spans
    topics: list[str] = Field(default_factory=list)  # fine-grained topics it groups


class ThemeListResponse(BaseModel):
    themes: list[Theme] = Field(default_factory=list)


class EntityCapture(BaseModel):
    capture_id: str
    title: str = ""
    summary: str = ""


class EntityDetailResponse(BaseModel):
    """A single entity's page: its captures + co-occurring entities to hop to."""
    entity: EntityRef
    captures: list[EntityCapture] = Field(default_factory=list)
    related: list[EntityRef] = Field(default_factory=list)


class LocationSummary(BaseModel):
    """A place where captures were made — a memory hub."""
    name: str
    locality: str = ""
    lat: float | None = None
    lon: float | None = None
    count: int = 0


class LocationListResponse(BaseModel):
    locations: list[LocationSummary] = Field(default_factory=list)


# ── Auth ────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str


# ── Error model ────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    code: int = 500
