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

# Controlled vocabulary of broad domain/category tags.
# These are assigned during topic extraction and used to create *weak*
# category-level links (e.g. Subaru ↔ Tesla = both 'automotive') that pure
# specific-topic matching would miss. Kept lowercase for case-insensitive match.
DOMAIN_TOPICS = {
    # vehicles & tech
    "automotive", "programming", "electronics", "computers", "mobile",
    "gaming", "photography",
    # home & living
    "appliances", "furniture", "home", "kitchen", "tools", "garden",
    # personal & style
    "fashion", "watches", "jewelry", "beauty", "health", "sports",
    # lifestyle & media
    "food", "travel", "music", "entertainment", "books", "movies",
    # knowledge & money
    "finance", "education", "office", "toys", "pets", "baby",
}

# Generic terms that appear across unrelated captures (e-commerce boilerplate,
# country names) and must NOT count as meaningful shared topics or entities.
# Without this filter, every Czech product page links to every other one via
# "online shopping" / "Czech Republic". Compared case-insensitively.
GENERIC_TERMS = {
    "e-commerce", "ecommerce", "online shopping", "shopping", "product page",
    "product listing", "listing", "website", "web page", "webpage", "online store",
    "price", "pricing", "product", "retail", "purchase", "buy", "shop",
    "czech republic", "czech", "czechia", "europe", "usa", "uk",
    "internet", "online", "app", "application", "user interface", "ui",
}


def _is_meaningful(name: str) -> bool:
    """True if a topic/entity name is specific enough to corroborate a link."""
    n = (name or "").strip().lower()
    return bool(n) and n not in GENERIC_TERMS and n not in DOMAIN_TOPICS

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

        # 1. Embedding similarity — WITH topic corroboration + domain tiering.
        #
        # Pure embedding similarity is noisy: all Czech e-shop product pages
        # (cars, watches, clothing) look alike because they share the same
        # "product page, price in Kč incl. VAT" structure. A Subaru listing is
        # 0.71 similar to a *watch*. So embedding score alone is unreliable.
        #
        # We classify each candidate by what kind of topics it shares:
        #   • shares a SPECIFIC topic  → STRONG link  (strength = similarity)
        #   • shares only a DOMAIN tag → WEAK  link  (strength ≈ 0.30, secondary)
        #   • shares nothing           → dropped, unless near-duplicate
        NEAR_DUPLICATE = 0.80  # Neo4j-scaled score; above this, always link
        WEAK_DOMAIN_STRENGTH = 0.30  # below UI primary threshold → "slabší vazby"
        try:
            rec = await session.run(
                Q.EMBEDDING_SIMILARITY,
                capture_id=capture_id,
                embedding=embedding,
                k=limit * 2,
                threshold=threshold,
            )
            async for row in rec:
                sim = row["similarity"]
                names = row.get("shared_names") or []
                # Specific = meaningful content topic (not generic, not a domain tag)
                specific = [n for n in names if _is_meaningful(n)]
                domain   = [n for n in names if n and n.lower() in DOMAIN_TOPICS]

                if specific:
                    # Genuine content overlap → strong semantic link
                    _merge(results, row, LinkType.embedding_similarity, sim)
                elif domain:
                    # Only a broad category in common → weak "smaller" link
                    _merge(results, row, LinkType.embedding_similarity, WEAK_DOMAIN_STRENGTH)
                elif sim >= NEAR_DUPLICATE:
                    # No shared topics but nearly identical → near-duplicate
                    _merge(results, row, LinkType.embedding_similarity, sim)
                else:
                    logger.debug(
                        "Skipping embedding link %s (sim=%.3f, no shared topics — structural coincidence)",
                        (row.get("title") or "")[:40], sim,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding similarity query failed: %s", exc)

        # 2. Topic overlap — require ≥ 2 shared *meaningful* topics
        # (generic terms like "online shopping" are filtered out first)
        try:
            rec = await session.run(
                Q.TOPIC_OVERLAP,
                capture_id=capture_id,
                limit=limit * 2,
            )
            async for row in rec:
                meaningful = [n for n in (row.get("shared_names") or []) if _is_meaningful(n)]
                if len(meaningful) < 2:
                    continue  # 1 or 0 specific topics is not a strong thematic link
                # 2 shared → 0.40,  5 shared → 1.0
                strength = min(1.0, len(meaningful) / 5)
                _merge(results, row, LinkType.topic_overlap, strength)
        except Exception as exc:
            logger.warning("Topic overlap query failed: %s", exc)

        # 3. Entity match — require a shared *specific* person/place.
        # Generic places like "Czech Republic" are filtered so that every
        # Czech screenshot doesn't link to every other one.
        try:
            rec = await session.run(
                Q.ENTITY_MATCH,
                capture_id=capture_id,
                limit=limit * 2,
            )
            async for row in rec:
                ents = [n for n in (row.get("shared_entities") or []) if _is_meaningful(n)]
                if not ents:
                    continue
                strength = min(1.0, len(ents) / 2)
                _merge(results, row, LinkType.entity_match, strength)
        except Exception as exc:
            logger.warning("Entity match query failed: %s", exc)

        # NOTE: Temporal proximity is intentionally skipped.
        # "Uploaded on the same day" is not a meaningful semantic connection,
        # especially when all captures come from one batch upload session.
        # It can be re-enabled once the dataset grows and date-based context
        # becomes genuinely informative.

    # Filter by minimum strength and cap results.
    # Floor is 0.25 so weak domain-level links (strength ~0.30) survive and
    # appear in the UI's collapsible "slabší vazby" (smaller connections) tier.
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
    """
    Insert or upgrade an entry in the accumulator dict.

    Priority order (highest = preferred):
        4 — embedding_similarity   (semantic content match)
        3 — topic_overlap          (shared topics)
        2 — entity_match           (shared people/places)
        1 — temporal_proximity     (fallback)

    Within the same priority level, higher strength wins.
    """
    _PRIORITY = {
        LinkType.embedding_similarity: 4,
        LinkType.topic_overlap: 3,
        LinkType.entity_match: 2,
        LinkType.temporal_proximity: 1,
    }

    cid = row["capture_id"]
    existing = acc.get(cid)

    if existing is None:
        pass  # will be set below
    else:
        ep = _PRIORITY.get(existing.link_type, 0)
        np = _PRIORITY.get(link_type, 0)
        # Keep existing if it has higher priority, or same priority + higher strength
        if ep > np or (ep == np and existing.strength >= strength):
            return

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
