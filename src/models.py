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


class CaptureDetailResponse(BaseModel):
    capture: CaptureNode
    connections: list[CaptureLink] = Field(default_factory=list)
    subject: SubjectInfo | None = None


class SearchResult(BaseModel):
    capture_id: str
    title: str
    summary: str
    score: float
    created_at: datetime


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


# ── Error model ────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    code: int = 500
