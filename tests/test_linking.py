"""
Test suite — Fáze 4 / linking engine.

Creates 10 synthetic CaptureNode objects and verifies:
  1. cosineSimilarity (mathematical correctness)
  2. calculateLinkStrength (composite scoring)
  3. findSimilarCaptures (mocked Neo4j, checks result shape + ordering)
  4. Link de-duplication and min-strength filtering
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.link_engine import (
    calculateLinkStrength,
    cosineSimilarity,
    findSimilarCaptures,
)
from src.models import CaptureNode, CaptureType, ExtractedAttributes, LinkType


# ── Fixtures: 10 synthetic captures ──────────────────────────────────────────

def _make_attrs(
    title: str,
    topics: list[str],
    people: list[str] | None = None,
    places: list[str] | None = None,
    sentiment: str = "neutral",
) -> ExtractedAttributes:
    return ExtractedAttributes(
        title=title,
        summary=f"Summary of {title}",
        raw_text="",
        capture_type=CaptureType.screenshot,
        topics=topics,
        people=people or [],
        places=places or [],
        sentiment=sentiment,
    )


CAPTURES: list[CaptureNode] = [
    CaptureNode(attributes=_make_attrs("Python basics",        ["python", "programming", "beginner"])),
    CaptureNode(attributes=_make_attrs("Python advanced",      ["python", "programming", "advanced", "decorators"])),
    CaptureNode(attributes=_make_attrs("Machine learning 101", ["python", "ml", "scikit-learn"])),
    CaptureNode(attributes=_make_attrs("Deep learning",        ["ml", "neural-networks", "pytorch"])),
    CaptureNode(attributes=_make_attrs("React hooks guide",    ["javascript", "react", "frontend"])),
    CaptureNode(attributes=_make_attrs("Vue.js tutorial",      ["javascript", "vue", "frontend"])),
    CaptureNode(attributes=_make_attrs("Elon Musk speech",     ["tech", "ev"], people=["Elon Musk"])),
    CaptureNode(attributes=_make_attrs("Tesla announcement",   ["ev", "cars"],  people=["Elon Musk"], places=["Palo Alto"])),
    CaptureNode(attributes=_make_attrs("San Francisco weather",["weather"],      places=["San Francisco"])),
    CaptureNode(attributes=_make_attrs("Palo Alto real estate",["realestate"],   places=["Palo Alto", "San Francisco"])),
]


# ── Tests: cosineSimilarity ───────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosineSimilarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosineSimilarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosineSimilarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_general_case(self):
        a = [1.0, 3.0, -5.0]
        b = [4.0, -2.0, -1.0]
        dot = 1 * 4 + 3 * (-2) + (-5) * (-1)   # = 4 - 6 + 5 = 3
        mag_a = math.sqrt(1 + 9 + 25)
        mag_b = math.sqrt(16 + 4 + 1)
        expected = dot / (mag_a * mag_b)
        assert cosineSimilarity(a, b) == pytest.approx(expected, abs=1e-9)

    def test_zero_vector_returns_0(self):
        assert cosineSimilarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosineSimilarity([1.0, 2.0], [1.0])


# ── Tests: calculateLinkStrength ──────────────────────────────────────────────

class TestCalculateLinkStrength:
    def test_identical_attributes_no_embedding(self):
        attrs = _make_attrs("Same", ["a", "b", "c"])
        strength = calculateLinkStrength(attrs, attrs)
        # topic Jaccard=1 (0.3) + sentiment match (0.1) = 0.4 minimum
        assert strength >= 0.4

    def test_disjoint_topics_no_shared_entities(self):
        a1 = _make_attrs("Python",     ["python"])
        a2 = _make_attrs("Marketing",  ["marketing", "seo"])
        strength = calculateLinkStrength(a1, a2)
        # topics disjoint = 0, entities empty = 0, sentiment match = 0.1
        assert strength == pytest.approx(0.1)

    def test_embedding_boosts_strength(self):
        a1 = _make_attrs("Doc A", ["x"])
        a2 = _make_attrs("Doc B", ["x"])
        emb_similar = [1.0] + [0.0] * 1023
        strength_with_emb = calculateLinkStrength(a1, a2, emb_similar, emb_similar)
        strength_without  = calculateLinkStrength(a1, a2)
        assert strength_with_emb > strength_without

    def test_shared_entities_increase_strength(self):
        a1 = _make_attrs("Doc A", [], people=["Alice", "Bob"])
        a2 = _make_attrs("Doc B", [], people=["Alice"])
        strength = calculateLinkStrength(a1, a2)
        # 1/2 entity Jaccard * 0.15 = 0.075 + sentiment 0.1
        assert strength == pytest.approx(0.175)

    def test_output_clamped_0_to_1(self):
        # Maximal overlap on all signals
        attrs = _make_attrs("X", ["a", "b"], people=["P"], places=["L"], sentiment="positive")
        emb = [1.0] + [0.0] * 1023
        strength = calculateLinkStrength(attrs, attrs, emb, emb)
        assert 0.0 <= strength <= 1.0


# ── Tests: findSimilarCaptures ────────────────────────────────────────────────

class TestFindSimilarCaptures:
    """Mock Neo4j session and verify shape + ordering of results."""

    @staticmethod
    def _mock_session_rows():
        """Return a list of dicts mimicking Neo4j record data."""
        return [
            {"capture_id": f"id-{i:02d}", "title": f"Capture {i}", "summary": f"Summary {i}",
             "similarity": 0.9 - i * 0.05, "shared_topics": 3, "entity_count": 2,
             "days_apart": 5}
            for i in range(10)
        ]

    @pytest.mark.asyncio
    async def test_returns_linked_captures(self):
        """findSimilarCaptures should return a list of LinkedCapture objects."""
        rows = self._mock_session_rows()

        # Create mock async record iterator
        async def _iter_rows(query, **kwargs):
            mock_result = MagicMock()

            async def __aiter__(self):
                for r in rows[:3]:  # limit results per query
                    yield r

            mock_result.__aiter__ = __aiter__
            return mock_result

        mock_session = AsyncMock()
        mock_session.run = _iter_rows
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        attrs = CAPTURES[0].attributes
        embedding = [0.1] * 1024

        with patch("src.link_engine.get_session", return_value=mock_session):
            results = await findSimilarCaptures("src-id", attrs, embedding)

        # Must be a list
        assert isinstance(results, list)
        # All items are LinkedCapture-like
        for item in results:
            assert hasattr(item, "capture_id")
            assert hasattr(item, "strength")
            assert hasattr(item, "link_type")
            assert 0.0 <= item.strength <= 1.0

    @pytest.mark.asyncio
    async def test_results_ordered_by_strength_desc(self):
        """Results must be sorted strongest-first."""
        rows = self._mock_session_rows()

        async def _iter_rows(query, **kwargs):
            mock_result = MagicMock()

            async def __aiter__(self):
                for r in rows:
                    yield r

            mock_result.__aiter__ = __aiter__
            return mock_result

        mock_session = AsyncMock()
        mock_session.run = _iter_rows
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        attrs = CAPTURES[0].attributes
        embedding = [0.1] * 1024

        with patch("src.link_engine.get_session", return_value=mock_session):
            results = await findSimilarCaptures("src-id", attrs, embedding)

        if len(results) > 1:
            strengths = [r.strength for r in results]
            assert strengths == sorted(strengths, reverse=True)

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """The same capture_id appearing in multiple queries should appear only once."""
        dup_rows = [
            {"capture_id": "dup-01", "title": "Dup", "summary": "s",
             "similarity": 0.9, "shared_topics": 3, "entity_count": 2, "days_apart": 2}
        ]

        async def _iter_dup(query, **kwargs):
            mock_result = MagicMock()

            async def __aiter__(self):
                for r in dup_rows:
                    yield r

            mock_result.__aiter__ = __aiter__
            return mock_result

        mock_session = AsyncMock()
        mock_session.run = _iter_dup
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        attrs = CAPTURES[0].attributes
        with patch("src.link_engine.get_session", return_value=mock_session):
            results = await findSimilarCaptures("src-id", attrs, [0.0] * 1024)

        ids = [r.capture_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate capture_ids in result"

    @pytest.mark.asyncio
    async def test_respects_min_strength(self):
        """Links below link_strength_min must be filtered out."""
        weak_rows = [
            {"capture_id": "weak-01", "title": "W", "summary": "s",
             "similarity": 0.01, "shared_topics": 0, "entity_count": 0, "days_apart": 89}
        ]

        async def _iter_weak(query, **kwargs):
            mock_result = MagicMock()

            async def __aiter__(self):
                for r in weak_rows:
                    yield r

            mock_result.__aiter__ = __aiter__
            return mock_result

        mock_session = AsyncMock()
        mock_session.run = _iter_weak
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        attrs = CAPTURES[0].attributes
        with patch("src.link_engine.get_session", return_value=mock_session):
            results = await findSimilarCaptures("src-id", attrs, [0.0] * 1024)

        # All results must satisfy the configured minimum strength
        from src.config import settings
        for r in results:
            assert r.strength >= settings.link_strength_min
