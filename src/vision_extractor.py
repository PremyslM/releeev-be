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
    # max_retries=0: on the free tier's 3 RPM limit, fail fast (~100ms) instead
    # of letting tenacity back-off for ~1.7s and blocking the search request.
    return voyageai.Client(api_key=key, max_retries=0)


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM = """You are an intelligent capture analyser for a second-brain app.
Look at the image and return a strict JSON object with these fields (all required):

{
  "title":        "<short, specific title of WHAT this is>",
  "summary":      "<the takeaway — see rules>",
  "raw_text":     "<readable text transcribed verbatim, or empty string>",
  "capture_type": "<one of: screenshot|photo|document|unknown>",
  "topics":       ["<keyword1>", ...],
  "category":     "<exactly ONE broad domain category from the allowed list>",
  "people":       ["<full name if visible>", ...],
  "places":       ["<city / building / URL if visible>", ...],
  "language":     "<ISO-639-1 code, e.g. en>",
  "sentiment":    "<positive|negative|neutral>",
  "confidence":   <float 0.0-1.0 representing your extraction confidence>
}

SUMMARY RULES — this is the important part:
- Do NOT describe what is visibly in the picture; the user already saw it.
  Never write "a photo of…", "the image shows…", "a screenshot of…".
- Instead, infer and state what it MEANS or what it's FOR: the point, the
  decision, the fact worth remembering, the conclusion. Be concise (1–3 sentences).
- If it identifies a real thing (a product, place, dish, document), name it and
  give the useful gist (e.g. what it is, key spec/price/where), not its looks.
- For a document/screenshot with text, summarise the substance, not the layout.

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
    from src.link_engine import DOMAIN_TOPICS  # local import: avoids any cycle

    t0 = time.perf_counter()
    client = _claude()
    allowed = ", ".join(sorted(DOMAIN_TOPICS))

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
                        "text": (
                            "Analyse this image and return the JSON object. "
                            f'The "category" must be exactly ONE of: {allowed}. '
                            "Pick the closest one even for photos (e.g. a car photo → "
                            "automotive). Do NOT put the category word itself into topics."
                        ),
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

    # Ensure the broad domain category is one of the topics, so photos link to
    # documents of the same domain (e.g. a car photo ↔ a car listing via
    # "automotive"). Mirrors the URL/text extraction path.
    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:15]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS and category not in [t.lower() for t in topics]:
        topics.append(category)

    return ExtractedAttributes(
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        raw_text=data.get("raw_text", ""),
        capture_type=ct,
        topics=topics,
        people=data.get("people", []),
        places=data.get("places", []),
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        confidence=float(data.get("confidence", 1.0)),
    )


def extractAttributesFromText(
    text: str, url: str = "", signals: dict | None = None
) -> ExtractedAttributes:
    """
    Build structured attributes from a web page's text (instead of an image).

    Used by POST /analyze-url. Mirrors extractAttributes() but the input is the
    page's text + HTML signals (title, OG tags) rather than pixels.
    """
    from src.link_engine import DOMAIN_TOPICS  # local import: avoids any cycle

    client = _claude()
    signals = signals or {}
    sig_lines = "\n".join(f"{k}: {v}" for k, v in signals.items() if v)
    allowed = ", ".join(sorted(DOMAIN_TOPICS))

    prompt = f"""Analyse this web page / video and return ONLY valid JSON, no prose.

URL: {url}
{sig_lines}

Content (truncated):
{text[:6000]}

Return JSON:
{{
  "title":    "<concise, specific title of what this is>",
  "summary":  "<a genuinely useful synthesis — see rules>",
  "topics":   ["<6-10 specific English topic keywords>"],
  "category": "<exactly ONE of: {allowed}>",
  "people":   ["<real person names, or []>"],
  "places":   ["<real place names / brands, or []>"],
  "language": "<ISO-639-1 code>",
  "sentiment":"<positive|negative|neutral>"
}}

SUMMARY RULES — extract what's worth keeping, don't just describe the page:
- Adapt the shape to the content:
  • Recipe → an "Ingredients:" bulleted list, then numbered "Steps:".
  • How-to / tutorial → the key steps or takeaways as a short list.
  • Product → what it is + key specs / price as a few bullets.
  • Article / video → 3-6 bullet key points or conclusions.
- Write it in the content's own language. Draw conclusions; be concrete and
  skimmable. Use plain bullets ("- ") and "1." numbering, short lines. Keep it
  tight — no fluff, no "this page shows…".

Topics: canonical/standardized; specific over generic; do NOT put the category
word itself into topics."""

    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next((b.text for b in resp.content if b.type == "text"), "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("URL attribute JSON parse failed: %s\nRaw: %s", exc, raw[:300])
        data = {}

    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:15]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS and category not in [t.lower() for t in topics]:
        topics.append(category)

    return ExtractedAttributes(
        title=data.get("title", "") or (signals.get("og:title") or signals.get("page title") or url),
        summary=data.get("summary", ""),
        raw_text=text[:2000],
        capture_type=CaptureType.document,
        topics=topics,
        people=[p for p in data.get("people", []) if isinstance(p, str)][:10],
        places=[pl for pl in data.get("places", []) if isinstance(pl, str)][:10],
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        confidence=0.85,
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


def getEmbeddingsBatch(texts: list[str]) -> list[list[float]]:
    """
    Embed many texts in a SINGLE Voyage API request.

    The Voyage free tier is limited to 3 requests/minute, so embedding captures
    one-by-one quickly hits the rate limit. Batching all texts into one call
    avoids that entirely (one request, regardless of how many texts).

    Returns a list of 1024-dim vectors aligned with the input order.
    Empty/blank inputs get a zero-vector; on total failure all are zero-vectors.
    """
    if not texts:
        return []

    # Replace blank texts with a single space (Voyage rejects empty strings)
    safe = [t if t.strip() else " " for t in texts]

    try:
        vc = _voyage()
        result = vc.embed(
            texts=safe,
            model="voyage-3",
            input_type="document",
        )
        out: list[list[float]] = []
        for i, emb in enumerate(result.embeddings):
            out.append([0.0] * 1024 if not texts[i].strip() else emb)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.error("getEmbeddingsBatch failed (%d texts): %s", len(texts), exc)
        return [[0.0] * 1024 for _ in texts]
