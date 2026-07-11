"""
Linking engine — similarity computation + graph-based capture matching.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from src import queries as Q
from src.config import settings
from src.models import ExtractedAttributes, LinkedCapture, LinkType
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


# ── Evidence accumulation model ────────────────────────────────────────────────
#
# Instead of letting each signal compete and keeping a single "winning" link
# type, we collect ALL evidence for each candidate pair and combine it into one
# confidence score. The key benefit: several *weak* signals add up to a *strong*
# link. Two captures that share only a category (weak) but were also taken at the
# same place and in the same session (future iOS signals) become a confident link
# — something no single-signal model could express.

@dataclass
class _Evidence:
    """All accumulated signals for one candidate capture (the 'other' side)."""
    capture_id: str
    title: str = ""
    summary: str = ""

    # Content signals (available today)
    embedding_sim: float = 0.0           # Neo4j vector score (0 if below threshold)
    specific_topics: list[str] = field(default_factory=list)  # meaningful shared topics
    shared_domain: str | None = None     # shared broad category (weak)
    shared_entities: list[str] = field(default_factory=list)  # shared person/place/brand

    # Contextual signals (filled once the iOS app captures geo / time / source).
    # Wired in here so adding them later is a data change, not an engine rewrite.
    same_place: bool = False             # captured at the same venue (~100 m)
    same_session: bool = False           # captured within minutes of each other
    same_source: bool = False            # same website domain / author


# Tunable weights. Each signal independently contributes to the 0..1 score.
# Calibrated so that:
#   • domain-only            → ~0.30  (weak / secondary tier in UI)
#   • ≥1 specific topic       → ≥0.40  (strong / primary tier)
#   • near-duplicate          → ~similarity (≈0.9)
#   • weak signals stack      → can cross into the strong tier together
_W = {
    "topic_base":   0.40, "topic_extra": 0.10, "topic_cap": 0.65,
    "entity_base":  0.30, "entity_extra": 0.10, "entity_cap": 0.55,
    "domain_only":  0.30, "domain_corrob": 0.10,
    "embedding":    0.20,
    "place":        0.30, "session": 0.20, "source": 0.25,  # future iOS signals
    "near_dup":     0.80,
}


# A link must be *earned* by a specific, meaningful signal — not by weak
# commonality alone (same broad category, same place/time/source). Otherwise
# everything links to everything and the graph becomes a meaningless hairball.
_MEANINGFUL_SEMANTIC = 0.72   # embedding this high ⇒ genuinely about the same thing


def _has_specific_anchor(ev: "_Evidence") -> bool:
    """True if the pair shares a concrete reason to be linked — a specific topic,
    a named entity/brand, or strong semantic similarity. Weak-only evidence
    (broad category, same place/time/source) does NOT qualify."""
    return bool(
        ev.specific_topics
        or ev.shared_entities
        or ev.embedding_sim >= _MEANINGFUL_SEMANTIC
    )


def _score_evidence(ev: _Evidence, threshold: float) -> tuple[float, LinkType, list[str]]:
    """
    Combine all of a candidate's evidence into (confidence, display_type, reasons).

    `reasons` is a human-readable explanation list shown in the UI ("proč").
    """
    score = 0.0
    reasons: list[str] = []

    # 1. Specific shared topics — strongest content signal
    if ev.specific_topics:
        n = len(ev.specific_topics)
        score += min(_W["topic_cap"], _W["topic_base"] + _W["topic_extra"] * (n - 1))
        reasons.append("témata: " + ", ".join(ev.specific_topics[:3]))

    # 2. Shared specific entities (person / brand / specific place)
    if ev.shared_entities:
        n = len(ev.shared_entities)
        score += min(_W["entity_cap"], _W["entity_base"] + _W["entity_extra"] * (n - 1))
        reasons.append("entity: " + ", ".join(ev.shared_entities[:3]))

    # 3. Shared broad category — weak on its own, light corroboration otherwise
    if ev.shared_domain:
        score += _W["domain_corrob"] if ev.specific_topics else _W["domain_only"]
        reasons.append(f"kategorie: {ev.shared_domain}")

    # 4. Embedding similarity — corroborating signal, scaled above the threshold
    if ev.embedding_sim > 0:
        norm = max(0.0, min(1.0, (ev.embedding_sim - threshold) / max(1e-6, 1.0 - threshold)))
        score += _W["embedding"] * norm
        if norm > 0.4:
            reasons.append("významová podobnost")

    # 5. Contextual signals from the iOS app (place / time / source)
    if ev.same_place:
        score += _W["place"];   reasons.append("stejné místo")
    if ev.same_session:
        score += _W["session"]; reasons.append("stejná chvíle")
    if ev.same_source:
        score += _W["source"];  reasons.append("stejný zdroj")

    # 6. Near-duplicate floor — almost-identical captures are strong on their own
    if ev.embedding_sim >= _W["near_dup"]:
        score = max(score, ev.embedding_sim)
        if not reasons:
            reasons.append("téměř duplikát")

    score = min(1.0, score)

    # Pick the dominant link type for the icon/label (display only)
    if ev.same_place or ev.same_session:
        display = LinkType.temporal_proximity        # contextual (place/time)
    elif ev.shared_entities:
        display = LinkType.entity_match
    elif ev.specific_topics:
        display = LinkType.topic_overlap
    else:
        display = LinkType.embedding_similarity      # domain-only / near-dup

    return score, display, reasons


# ── Graph queries ─────────────────────────────────────────────────────────────

async def findSimilarCaptures(
    capture_id: str,
    attributes: ExtractedAttributes,
    embedding: list[float],
    *,
    limit: int = 10,
) -> list[LinkedCapture]:
    """
    Gather evidence from every available signal, combine it per candidate into a
    single confidence score, and return the strongest links.

    This is the 'evidence accumulation' model: signals don't compete (as in the
    old priority-based merge) — they add up. A pair sharing only a category is a
    weak link; the same pair sharing a category AND captured together becomes a
    strong one.
    """
    threshold = settings.cosine_similarity_threshold
    evidence: dict[str, _Evidence] = {}

    def _ev(cid: str, row) -> _Evidence:
        e = evidence.get(cid)
        if e is None:
            e = _Evidence(
                capture_id=cid,
                title=row.get("title", "") or "",
                summary=row.get("summary", "") or "",
            )
            evidence[cid] = e
        return e

    async with get_session() as session:

        # ── Signal A: embedding similarity (vector index) ───────────────────────
        try:
            rec = await session.run(
                Q.EMBEDDING_SIMILARITY,
                capture_id=capture_id,
                embedding=embedding,
                k=limit * 3,
                threshold=threshold,
            )
            async for row in rec:
                e = _ev(row["capture_id"], row)
                e.embedding_sim = float(row["similarity"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding similarity query failed: %s", exc)

        # ── Signal B: shared topics (specific) + shared domain (category) ───────
        # TOPIC_OVERLAP returns every pair sharing ≥1 topic, including pairs whose
        # embedding is below threshold — so this is the authoritative topic source.
        try:
            rec = await session.run(
                Q.TOPIC_OVERLAP,
                capture_id=capture_id,
                limit=limit * 4,
            )
            async for row in rec:
                names = row.get("shared_names") or []
                specific = [n for n in names if _is_meaningful(n)]
                domain   = next((n.lower() for n in names if n and n.lower() in DOMAIN_TOPICS), None)
                if not specific and not domain:
                    continue
                e = _ev(row["capture_id"], row)
                e.specific_topics = specific
                if domain:
                    e.shared_domain = domain
        except Exception as exc:
            logger.warning("Topic overlap query failed: %s", exc)

        # ── Signal C: shared specific entities (person / place / brand) ─────────
        try:
            rec = await session.run(
                Q.ENTITY_MATCH,
                capture_id=capture_id,
                limit=limit * 3,
            )
            async for row in rec:
                ents = [n for n in (row.get("shared_entities") or []) if _is_meaningful(n)]
                if not ents:
                    continue
                e = _ev(row["capture_id"], row)
                e.shared_entities = ents
        except Exception as exc:
            logger.warning("Entity match query failed: %s", exc)

        # ── Signal D: same venue (co-located memories) ──────────────────────────
        # Captures made at the same place — "that afternoon at café 14g". Feeds
        # the same_place term in the scoring function.
        try:
            rec = await session.run(
                Q.LOCATION_MATCH,
                capture_id=capture_id,
                limit=limit * 2,
            )
            async for row in rec:
                e = _ev(row["capture_id"], row)
                e.same_place = True
        except Exception as exc:
            logger.warning("Location match query failed: %s", exc)

        # ── Signals E/F: session / source (future) ──────────────────────────────
        # When timestamps (bursts) and shared-URL metadata are captured, populate
        # e.same_session / e.same_source here; the scoring already accounts for them.

    # Combine evidence → confidence score per candidate
    min_strength = settings.link_strength_min
    links: list[LinkedCapture] = []
    for e in evidence.values():
        # Gate: skip links that rest ONLY on weak commonality (same category /
        # place / time / source). A link must have a concrete anchor so the graph
        # shows genuine, intelligible connections instead of a hairball.
        if not _has_specific_anchor(e):
            continue
        score, display_type, reasons = _score_evidence(e, threshold)
        if score < min_strength:
            continue
        links.append(
            LinkedCapture(
                capture_id=e.capture_id,
                link_type=display_type,
                strength=round(score, 4),
                title=e.title,
                summary=e.summary,
                reasons=reasons,
            )
        )

    # Keep only each capture's strongest, most-specific links — no node becomes a
    # hub of dozens of weak edges.
    links.sort(key=lambda x: x.strength, reverse=True)
    return links[: settings.max_links_per_capture]


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
                reasons=link.reasons,
            )
    logger.info("Persisted %d links from %s", len(links), source_id)
