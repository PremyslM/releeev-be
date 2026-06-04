"""
Test suite — Fáze 4 / extraction.

Tests three image types:
  1. Code screenshot  (synthetic base64 PNG)
  2. Book page        (text-heavy image)
  3. Product photo    (e-commerce style)

Each test verifies that the returned JSON schema matches ExtractedAttributes
and that required fields are present and correctly typed.

Run:
    pytest tests/test_extraction.py -v
"""
from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.models import CaptureType, ExtractedAttributes
from src.vision_extractor import extractAttributes, getEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image_b64(text: str, width: int = 400, height: int = 300) -> str:
    """Generate a tiny synthetic PNG with `text` drawn on it, returned as base64."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Fixtures ──────────────────────────────────────────────────────────────────

CODE_IMG   = _make_image_b64("def fibonacci(n):\n    return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)")
BOOK_IMG   = _make_image_b64("Chapter 1: The Origins of Knowledge\n\nKnowledge has been a fundamental pursuit...")
PRODUCT_IMG = _make_image_b64("iPhone 15 Pro Max\n$1,199\n★★★★★ (4,210 reviews)")


def _mock_claude_response(raw_json: str):
    """Return a mocked Anthropic response containing the given JSON text."""
    block = MagicMock()
    block.type = "text"
    block.text = raw_json

    response = MagicMock()
    response.content = [block]
    return response


# ── Tests: schema validation ──────────────────────────────────────────────────

class TestExtractAttributesSchema:
    """Verify that extractAttributes returns a valid ExtractedAttributes for each image type."""

    @pytest.mark.parametrize(
        "image_b64, raw_json, expected_type",
        [
            (
                CODE_IMG,
                """{
                    "title": "Fibonacci function in Python",
                    "summary": "A Python code snippet implementing the Fibonacci sequence recursively.",
                    "raw_text": "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
                    "capture_type": "screenshot",
                    "topics": ["python", "recursion", "algorithms"],
                    "people": [],
                    "places": [],
                    "language": "en",
                    "sentiment": "neutral",
                    "confidence": 0.98
                }""",
                CaptureType.screenshot,
            ),
            (
                BOOK_IMG,
                """{
                    "title": "Chapter 1: The Origins of Knowledge",
                    "summary": "Opening chapter discussing the fundamental pursuit of knowledge throughout history.",
                    "raw_text": "Chapter 1: The Origins of Knowledge\\n\\nKnowledge has been a fundamental pursuit...",
                    "capture_type": "document",
                    "topics": ["knowledge", "philosophy", "history"],
                    "people": [],
                    "places": [],
                    "language": "en",
                    "sentiment": "positive",
                    "confidence": 0.92
                }""",
                CaptureType.document,
            ),
            (
                PRODUCT_IMG,
                """{
                    "title": "iPhone 15 Pro Max Product Listing",
                    "summary": "Product photo showing iPhone 15 Pro Max priced at $1,199 with a 5-star rating.",
                    "raw_text": "iPhone 15 Pro Max $1,199 4,210 reviews",
                    "capture_type": "photo",
                    "topics": ["apple", "iphone", "smartphone", "tech"],
                    "people": [],
                    "places": [],
                    "language": "en",
                    "sentiment": "positive",
                    "confidence": 0.96
                }""",
                CaptureType.photo,
            ),
        ],
        ids=["code_screenshot", "book_page", "product_photo"],
    )
    def test_schema_valid(self, image_b64, raw_json, expected_type):
        """extractAttributes should return a properly-typed ExtractedAttributes."""
        with patch(
            "src.vision_extractor._claude",
            return_value=MagicMock(
                messages=MagicMock(
                    create=MagicMock(return_value=_mock_claude_response(raw_json))
                )
            ),
        ):
            result = extractAttributes(image_b64)

        # Type
        assert isinstance(result, ExtractedAttributes)

        # Required string fields
        assert isinstance(result.title,   str) and result.title
        assert isinstance(result.summary, str) and result.summary

        # Capture type
        assert result.capture_type == expected_type

        # Lists
        assert isinstance(result.topics, list)
        assert isinstance(result.people, list)
        assert isinstance(result.places, list)

        # Numeric / bounded fields
        assert 0.0 <= result.confidence <= 1.0
        assert result.sentiment in ("positive", "negative", "neutral")

    def test_malformed_json_returns_defaults(self):
        """Malformed JSON from Claude should degrade gracefully to default values."""
        with patch(
            "src.vision_extractor._claude",
            return_value=MagicMock(
                messages=MagicMock(
                    create=MagicMock(
                        return_value=_mock_claude_response("not valid json {{{")
                    )
                )
            ),
        ):
            result = extractAttributes(CODE_IMG)

        assert isinstance(result, ExtractedAttributes)
        assert result.capture_type == CaptureType.unknown

    def test_empty_topics_list(self):
        """An image with no discernible topics should return an empty list, not error."""
        raw_json = """{
            "title": "Blank image",
            "summary": "Empty white image.",
            "raw_text": "",
            "capture_type": "photo",
            "topics": [],
            "people": [],
            "places": [],
            "language": "en",
            "sentiment": "neutral",
            "confidence": 0.5
        }"""
        with patch(
            "src.vision_extractor._claude",
            return_value=MagicMock(
                messages=MagicMock(
                    create=MagicMock(return_value=_mock_claude_response(raw_json))
                )
            ),
        ):
            result = extractAttributes(CODE_IMG)

        assert result.topics == []


# ── Tests: embedding ──────────────────────────────────────────────────────────

class TestGetEmbedding:
    def test_returns_list_of_floats(self):
        mock_emb = [0.1, -0.2, 0.3] * 342  # 1026 items — use first 1024
        mock_emb = mock_emb[:1024]

        with patch(
            "src.vision_extractor._voyage",
            return_value=MagicMock(
                embed=MagicMock(
                    return_value=MagicMock(embeddings=[mock_emb])
                )
            ),
        ):
            result = getEmbedding("Python recursion algorithms")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    def test_empty_text_returns_zero_vector(self):
        result = getEmbedding("   ")
        assert result == [0.0] * 1024

    def test_voyage_failure_returns_zero_vector(self):
        with patch(
            "src.vision_extractor._voyage",
            return_value=MagicMock(
                embed=MagicMock(side_effect=RuntimeError("connection error"))
            ),
        ):
            result = getEmbedding("some text")

        assert result == [0.0] * 1024
