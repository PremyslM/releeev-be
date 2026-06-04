"""
Linking engine — similarity computation + graph-based capture matching.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from src import queries as Q
from src.config import settings
from src.models import CaptureLink, ExtractedAttributes, LinkedCapture, LinkType
from src.neo4j_driver import get_session

logger = logging.getLogger(__name__)

# ── Math helpers ──────────────────────────────────────────────────────────────

def cosineSimilarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Pure-Python cosine similarity.

    Returns a value in [-1, 1]; 1.0 means identical direction.
    Returns 0.0 if either vector is a zero-vector.
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector length mismatch: {len(vec1)} vs {len(vec2)}"
        )

    dot  = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return dot / (mag1 * mag2)


# ── Link strength ─────────────────────────────────────────────────────────────

def calculateLinkStrength(
    attrs1: ExtractedAttributes,
    attrs2: ExtractedAttributes,
    emb1: list[float] | None = None,
    emb2: list[float] | None = None,
) -> float:
    """
    Weighted aggregate of four signals → 0.0–1.0.

    Weights (sum to 1):
        embedding similarity  0.45
        topic overlap         0.30
        entity overlap        0.15
        sentiment match       0.10
    """
    score = 0.0

    # 1. Embedding similarity (cosine)
    if emb1 and emb2 and any(emb1) and any(emb2):
        cos = cosineSimilarity(emb1, emb2)
        score += max(0.0, cos) * 0.45

    # 2. Topic overlap (Jaccard)
    t1, t2 = set(attrs1.topics), set(attrs2.topics)
    if t1 or t2:
        jaccard = len(t1 & t2) / len(t1 | t2)
        score += jaccard * 0.30

    # 3. Entity overlap (Jaccard over people + places)
    e1 = set(attrs1.people) | set(attrs1.places)
    e2 = set(attrs2.people) | set(attrs2.places)
    if e1 or e2:
        jaccard_e = len(e1 & e2) / len(e1 | e2)
        score += jaccard_e * 0.15

    # 4. Sentiment match
    if attrs1.sentiment == attrs2.sentiment:
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)


# ── Graph queries ─────────────────────────────────────────────────────────────

async def findSimilarCaptures(
    capture_id: str,
    attributes: ExtractedAttributes,
    embedding: list[float],
    *,
    limit: int = 10,
) -> list[LinkedCapture]:
    """
    Run all four Cypher similarity queries in sequence and merge results,
    de-duplicating by capture_id and keeping the highest-strength link per pair.
    """
    threshold = settings.cosine_similarity_threshold
    days      = settings.temporal_proximity_days
    results: dict[str, LinkedCapture] = {}

    async with get_session() as session:

        # 1. Embedding similarity (Neo4j vector index)
        try:
            rec = await session.run(
                Q.EMBEDDING_SIMILARITY,
                capture_id=capture_id,
                embedding=embedding,
                k=limit * 2,
                threshold=threshold,
            )
            async for row in rec:
                _merge(results, row, LinkType.embedding_similarity, row["similarity"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding similarity query failed: %s", exc)

        # 2. Topic overlap
        try:
            rec = await session.run(
                Q.TOPIC_OVERLAP,
                capture_id=capture_id,
                min_shared=1,
                limit=limit * 2,
            )
            async for row in rec:
                strength = min(1.0, row["shared_topics"] / 10)
                _merge(results, row, LinkType.topic_overlap, strength)
        except Exception as exc:
            logger.warning("Topic overlap query failed: %s", exc)

        # 3. Entity match
        try:
            rec = await session.run(
                Q.ENTITY_MATCH,
                capture_id=capture_id,
                limit=limit * 2,
            )
            async for row in rec:
                strength = min(1.0, row["entity_count"] / 5)
                _merge(results, row, LinkType.entity_match, strength)
        except Exception as exc:
            logger.warning("Entity match query failed: %s", exc)

        # 4. Temporal proximity
        try:
            rec = await session.run(
                Q.TEMPORAL_PROXIMITY,
                capture_id=capture_id,
                days=days,
                limit=limit * 2,
            )
            async for row in rec:
                strength = max(0.0, 1.0 - row["days_apart"] / days)
                _merge(results, row, LinkType.temporal_proximity, strength)
        except Exception as exc:
            logger.warning("Temporal proximity query failed: %s", exc)

    # Filter by minimum strength and cap results
    min_strength = settings.link_strength_min
    sorted_links = sorted(
        (v for v in results.values() if v.strength >= min_strength),
        key=lambda x: x.strength,
        reverse=True,
    )
    return sorted_links[:limit]


def _merge(
    acc: dict[str, LinkedCapture],
    row: Any,
    link_type: LinkType,
    strength: float,
) -> None:
    """Insert or upgrade an entry in the accumulator dict."""
    cid = row["capture_id"]
    existing = acc.get(cid)
    if existing is None or strength > existing.strength:
        acc[cid] = LinkedCapture(
            capture_id=cid,
            link_type=link_type,
            strength=round(strength, 4),
            title=row.get("title", ""),
            summary=row.get("summary", ""),
        )


# ── Persist links ─────────────────────────────────────────────────────────────

async def persistLinks(
    source_id: str,
    links: list[LinkedCapture],
) -> None:
    """Write LINKED_TO relationships back to Neo4j."""
    if not links:
        return
    async with get_session() as session:
        for link in links:
            await session.run(
                Q.CREATE_LINK,
                source_id=source_id,
                target_id=link.capture_id,
                link_type=link.link_type.value,
                strength=link.strength,
            )
    logger.info("Persisted %d links from %s", len(links), source_id)
