"""
Claude Vision extraction + Voyage AI embeddings.

Public API:
    extractAttributes(image_base64, mime_type) -> ExtractedAttributes
    getEmbedding(text)                         -> list[float]  (1024-dim)
"""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache

import anthropic
import voyageai

from src.config import settings
from src.models import CaptureType, ExtractedAttributes

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache(maxsize=1)
def _voyage() -> voyageai.Client:
    # Use VOYAGE_API_KEY if set, otherwise try ANTHROPIC_API_KEY
    key = settings.voyage_api_key or settings.anthropic_api_key
    return voyageai.Client(api_key=key)


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM = """You are an intelligent capture analyser for a second-brain app.
Your job: look at the provided image and return a strict JSON object with the
following fields (all required):

{
  "title":        "<short descriptive title>",
  "summary":      "<2-3 sentence summary of the content>",
  "raw_text":     "<all readable text extracted verbatim, or empty string>",
  "capture_type": "<one of: screenshot|photo|document|unknown>",
  "topics":       ["<keyword1>", ...],
  "people":       ["<full name if visible>", ...],
  "places":       ["<city / building / URL if visible>", ...],
  "language":     "<ISO-639-1 code, e.g. en>",
  "sentiment":    "<positive|negative|neutral>",
  "confidence":   <float 0.0-1.0 representing your extraction confidence>
}

Respond with ONLY valid JSON — no markdown fences, no prose.
"""


# ── Main functions ────────────────────────────────────────────────────────────

def extractAttributes(
    image_base64: str,
    mime_type: str = "image/jpeg",
) -> ExtractedAttributes:
    """
    Send an image to Claude claude-opus-4-8 via the Vision API and parse the
    structured JSON response into an ExtractedAttributes object.
    """
    t0 = time.perf_counter()
    client = _claude()

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyse this image and return the JSON object.",
                    },
                ],
            }
        ],
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("extractAttributes completed in %.0f ms", elapsed)

    # Extract the text block (skip thinking blocks)
    raw_json = next(
        (block.text for block in response.content if block.type == "text"),
        "{}",
    )

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s\nRaw: %s", exc, raw_json[:300])
        data = {}

    # Map capture_type to enum safely
    ct_raw = data.get("capture_type", "unknown")
    try:
        ct = CaptureType(ct_raw)
    except ValueError:
        ct = CaptureType.unknown

    return ExtractedAttributes(
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        raw_text=data.get("raw_text", ""),
        capture_type=ct,
        topics=data.get("topics", []),
        people=data.get("people", []),
        places=data.get("places", []),
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        confidence=float(data.get("confidence", 1.0)),
    )


def getEmbedding(text: str) -> list[float]:
    """
    Generate a 1024-dimensional embedding vector using Voyage AI
    (voyage-3 model — Anthropic's first-party embedding service).

    Falls back to a zero-vector on error so downstream code stays functional.
    """
    if not text.strip():
        return [0.0] * 1024

    try:
        vc = _voyage()
        result = vc.embed(
            texts=[text],
            model="voyage-3",
            input_type="document",
        )
        return result.embeddings[0]
    except Exception as exc:  # noqa: BLE001
        logger.error("getEmbedding failed: %s", exc)
        return [0.0] * 1024
