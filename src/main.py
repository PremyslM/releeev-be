"""
Brain Graph — FastAPI application.

Endpoints:
    POST /analyze-capture
    GET  /captures/{id}
    GET  /search?q=...
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache

import anthropic
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import queries as Q
from src.config import settings
from src.link_engine import findSimilarCaptures, persistLinks
from src.models import (
    AnalyzeCaptureRequest,
    AnalyzeCaptureResponse,
    CaptureDetailResponse,
    CaptureLink,
    CaptureNode,
    ErrorResponse,
    ExtractedAttributes,
    LinkType,
    SearchResponse,
    SearchResult,
    SubjectInfo,
    SubjectSibling,
)
from src.neo4j_driver import close_driver, get_session
from src.schema_setup import run_schema_setup
from src.vision_extractor import extractAttributes, getEmbedding

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.app_log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brain_graph")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Brain Graph API (env=%s)", settings.app_env)
    await run_schema_setup()
    yield
    await close_driver()
    logger.info("Brain Graph API shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Graph API",
    description="Knowledge Linking Backend — captures, embeddings, graph links",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(anthropic.APIError)
async def anthropic_error_handler(request, exc: anthropic.APIError):
    logger.error("Anthropic API error: %s", exc)
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(
            error="Claude API error",
            detail=str(exc),
            code=502,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code=500,
        ).model_dump(),
    )


# ── Helper: persist a full capture to Neo4j ───────────────────────────────────

async def _persist_capture(capture: CaptureNode) -> None:
    attrs = capture.attributes
    async with get_session() as session:
        # Upsert Capture node
        await session.run(
            Q.UPSERT_CAPTURE,
            id=capture.id,
            title=attrs.title,
            summary=attrs.summary,
            raw_text=attrs.raw_text,
            image_hash=capture.image_hash,
            language=attrs.language,
            sentiment=attrs.sentiment,
            confidence=attrs.confidence,
            created_at=capture.created_at.isoformat(),
            embedding=capture.embedding,
        )
        # Topics
        for topic in attrs.topics:
            await session.run(Q.UPSERT_TOPIC, name=topic, capture_id=capture.id)
        # People
        for person in attrs.people:
            await session.run(Q.UPSERT_PERSON, name=person, capture_id=capture.id)
        # Places
        for place in attrs.places:
            await session.run(Q.UPSERT_PLACE, name=place, capture_id=capture.id)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/analyze-capture",
    response_model=AnalyzeCaptureResponse,
    summary="Upload image, extract attributes, find links",
)
async def analyze_capture(req: AnalyzeCaptureRequest) -> AnalyzeCaptureResponse:
    """
    1. Extract attributes via Claude Vision.
    2. Generate embedding via Voyage AI.
    3. Persist the new Capture + its entities to Neo4j.
    4. Find similar existing captures via 4 Cypher queries.
    5. Persist LINKED_TO relationships.
    6. Return capture_id, attributes and linked captures.
    """
    t0 = time.perf_counter()

    # ── Deduplication — check if this exact image was already uploaded ────────
    image_hash = hashlib.sha256(req.image_base64.encode()).hexdigest()
    async with get_session() as session:
        dup_result = await session.run(Q.GET_CAPTURE_BY_HASH, image_hash=image_hash)
        dup_record = await dup_result.single()
        if dup_record is not None:
            node = dup_record["c"]
            logger.info("Duplicate image detected (hash=%s), returning existing capture", image_hash[:12])
            attrs = ExtractedAttributes(
                title=node.get("title", ""),
                summary=node.get("summary", ""),
                raw_text=node.get("raw_text", ""),
                language=node.get("language", "en"),
                sentiment=node.get("sentiment", "neutral"),
                confidence=node.get("confidence", 1.0),
            )
            # Return existing capture — no Claude call, no Neo4j write
            return AnalyzeCaptureResponse(
                capture_id=node["id"],
                attributes=attrs,
                embedding_dim=len(node.get("embedding") or []),
                linked_to=[],
                processing_ms=int((time.perf_counter() - t0) * 1000),
            )

    # ── Extract ───────────────────────────────────────────────────────────────
    try:
        attributes: ExtractedAttributes = extractAttributes(
            req.image_base64, req.image_mime
        )
    except anthropic.AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Claude API rate limit exceeded") from exc
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc}") from exc

    # ── Embed ─────────────────────────────────────────────────────────────────
    embed_text = f"{attributes.title} {attributes.summary} {' '.join(attributes.topics)}"
    embedding = getEmbedding(embed_text)

    # ── Build Capture node ────────────────────────────────────────────────────
    capture = CaptureNode(
        image_hash=image_hash,
        attributes=attributes,
        embedding=embedding,
    )

    # ── Persist to Neo4j ──────────────────────────────────────────────────────
    try:
        await _persist_capture(capture)
    except Exception as exc:
        logger.error("Neo4j persist error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    # ── Find similar captures ─────────────────────────────────────────────────
    try:
        linked = await findSimilarCaptures(
            capture.id, attributes, embedding
        )
        await persistLinks(capture.id, linked)
    except Exception as exc:
        logger.warning("Link engine error (non-fatal): %s", exc)
        linked = []

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return AnalyzeCaptureResponse(
        capture_id=capture.id,
        attributes=attributes,
        embedding_dim=len(embedding),
        linked_to=linked,
        processing_ms=elapsed_ms,
    )


@app.get(
    "/captures/{capture_id}",
    response_model=CaptureDetailResponse,
    summary="Get a capture and all its connections",
)
async def get_capture(capture_id: str) -> CaptureDetailResponse:
    """Retrieve a single Capture node + all LINKED_TO relationships."""
    async with get_session() as session:
        # Fetch capture
        result = await session.run(Q.GET_CAPTURE_BY_ID, id=capture_id)
        record = await result.single()
        if record is None:
            raise HTTPException(status_code=404, detail=f"Capture '{capture_id}' not found")

        node = record["c"]

        # Parse attributes — topics/people/places come from related nodes in graph
        attrs = ExtractedAttributes(
            title=node.get("title", ""),
            summary=node.get("summary", ""),
            raw_text=node.get("raw_text", ""),
            language=node.get("language", "en"),
            sentiment=node.get("sentiment", "neutral"),
            confidence=node.get("confidence", 1.0),
            topics=[t for t in (record.get("topics") or []) if t],
            people=[p for p in (record.get("people") or []) if p],
            places=[pl for pl in (record.get("places") or []) if pl],
        )
        capture = CaptureNode(
            id=node["id"],
            image_hash=node.get("image_hash", ""),
            attributes=attrs,
            embedding=list(node.get("embedding") or []),
        )

        # Fetch connections with shared topics and other capture info
        conn_result = await session.run(Q.GET_CAPTURE_CONNECTIONS, id=capture_id)
        raw_connections: list[CaptureLink] = []
        async for row in conn_result:
            try:
                lt = LinkType(row.get("link_type") or "TOPIC_OVERLAP")
            except ValueError:
                lt = LinkType.topic_overlap

            shared = [t for t in (row.get("shared_topics") or []) if t]

            raw_connections.append(
                CaptureLink(
                    source_id=capture_id,
                    target_id=row.get("other_id") or "",
                    link_type=lt,
                    strength=float(row.get("strength") or 0.0),
                    other_title=row.get("other_title") or "",
                    other_summary=row.get("other_summary") or "",
                    shared_topics=shared,
                    reasons=[r for r in (row.get("reasons") or []) if r],
                )
            )

        # No post-filtering needed: the evidence-accumulation engine already
        # produces honest, comparable strengths. The UI tiers links into
        # "silné / slabší" by strength, so weak links are simply ranked lower.
        connections = sorted(raw_connections, key=lambda c: c.strength, reverse=True)

        # Subject branch: is this capture one source of a grouped real-world thing?
        subject = None
        try:
            subj_result = await session.run(Q.GET_CAPTURE_SUBJECT, id=capture_id)
            subj_rec = await subj_result.single()
            if subj_rec is not None:
                siblings = [
                    SubjectSibling(
                        id=s["id"],
                        title=s.get("title") or "",
                        source=s.get("source") or "",
                    )
                    for s in (subj_rec.get("siblings") or [])
                    if s and s.get("id") != capture_id  # exclude self
                ]
                subject = SubjectInfo(
                    subject_id=subj_rec["subject_id"],
                    name=subj_rec.get("subject_name") or "",
                    kind=subj_rec.get("subject_kind") or "",
                    siblings=siblings,
                )
        except Exception as exc:
            logger.debug("Subject lookup failed for %s: %s", capture_id[:8], exc)

    return CaptureDetailResponse(capture=capture, connections=connections, subject=subject)


def _expand_query_with_claude(q: str) -> str:
    """
    Use Claude to translate and expand a short search query.
    'hra' → 'hra game gaming video game play entertainment'
    Falls back to original query on any error.
    """
    if len(q.split()) >= 4:
        # Long queries don't need expansion
        return q
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=80,
            system=(
                "You expand short search queries for a knowledge graph. "
                "Return 4-6 English keywords/phrases separated by commas. "
                "No explanations, no sentences — only keywords."
            ),
            messages=[{"role": "user", "content": f"Expand: {q}"}],
        )
        expanded_text = next(
            (b.text.strip() for b in response.content if b.type == "text"), ""
        )
        combined = f"{q} {expanded_text.replace(',', ' ')}"
        logger.info("Query expanded: %r → %r", q, combined[:100])
        return combined
    except Exception as exc:
        logger.warning("Query expansion failed (using original): %s", exc)
        return q


# ── Cached query helpers (the two slow network calls) ────────────────────────
# Both are memoized: repeated/refined searches for the same text return instantly
# instead of paying for another Claude / Voyage round-trip.

@lru_cache(maxsize=512)
def _expand_query_cached(q: str) -> str:
    return _expand_query_with_claude(q)


@lru_cache(maxsize=512)
def _embed_query_cached(q: str) -> tuple[float, ...]:
    """voyage-3 is multilingual — a Czech query embeds into the same space as
    the English documents, so no translation is needed for semantic search."""
    from src.vision_extractor import _voyage
    vc = _voyage()
    result = vc.embed(texts=[q], model="voyage-3", input_type="query")
    return tuple(result.embeddings[0])


def _to_native(dt):
    return dt.to_native() if hasattr(dt, "to_native") else dt


# Reciprocal Rank Fusion — combines several ranked lists into one robust ranking
# without needing comparable score scales. score(d) = Σ w / (k + rank(d)).
_RRF_K = 60
_RRF_WEIGHTS = {"semantic": 1.0, "fulltext": 0.7, "topic": 1.3}
_SEMANTIC_MIN = 0.45     # voyage query↔doc floor; below this is noise, not a match
_MIN_RESULT_SCORE = 0.40  # drop results scoring below 40% of the top match


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid search (semantic + fulltext + topic) with rank fusion",
)
async def search_captures(
    q: str = Query(..., min_length=1, description="Search query (any language)"),
    limit: int = Query(default=20, ge=1, le=100),
    mode: str = Query(default="hybrid", description="'hybrid' | 'semantic' | 'fulltext'"),
    expand: bool = Query(default=True, description="Use Claude to expand/translate query"),
) -> SearchResponse:
    """
    Fast multilingual hybrid search:
      1. Query embedding (Voyage, multilingual) and Claude expansion run in
         PARALLEL, and both are cached — repeat searches are instant.
      2. Semantic, fulltext and topic queries run CONCURRENTLY against Neo4j.
      3. Results are merged with Reciprocal Rank Fusion (rank-based, robust to
         incompatible score scales) for stable, precise ordering.
    """
    loop = asyncio.get_running_loop()
    want_semantic = mode in ("semantic", "hybrid")
    want_fulltext = mode in ("fulltext", "hybrid")

    # Kick off the query embedding immediately (cached; multilingual)
    embed_fut = loop.run_in_executor(None, _embed_query_cached, q) if want_semantic else None

    meta: dict[str, dict] = {}          # cid → {title, summary, created_at}
    ranked: dict[str, list[str]] = {}   # source → ordered capture_ids

    def _keywords(text: str) -> list[str]:
        return list({w.strip() for w in text.replace(",", " ").split() if len(w.strip()) >= 3})

    async def _topic(keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        ids: list[str] = []
        async with get_session() as s:
            rec = await s.run(Q.TOPIC_BOOST_SEARCH, keywords=keywords, limit=limit * 2)
            async for row in rec:
                cid = row["capture_id"]
                meta.setdefault(cid, {"title": row["title"] or "", "summary": row["summary"] or "",
                                      "created_at": _to_native(row["created_at"])})
                ids.append(cid)
        return ids

    # ── Conditional Claude expansion (only when actually needed) ──────────────
    # English / topic-matching queries hit a Topic directly, so we skip Claude
    # entirely (fast). Only a short query that matches NO topic — typically a
    # cross-lingual miss like "programování" — pays for expansion+translation.
    if want_fulltext:
        topic_ids = await _topic(_keywords(q))
        if expand and not topic_ids and len(q.split()) < 4:
            expanded = await loop.run_in_executor(None, _expand_query_cached, q)
            if expanded != q:
                topic_ids = await _topic(_keywords(expanded))
        ranked["topic"] = topic_ids

    query_embedding: list[float] = []
    if embed_fut is not None:
        try:
            query_embedding = list(await embed_fut)
        except Exception as exc:
            # e.g. Voyage rate limit — degrade gracefully to fulltext + topic
            logger.warning("Query embedding unavailable, semantic skipped: %s", exc)

    # ── Concurrent Neo4j queries (semantic + fulltext) ────────────────────────
    async def _semantic() -> list[str]:
        ids: list[str] = []
        async with get_session() as s:
            rec = await s.run(Q.SEMANTIC_SEARCH, embedding=query_embedding,
                              k=limit * 3, limit=limit * 2)
            async for row in rec:
                if float(row["score"]) < _SEMANTIC_MIN:
                    continue  # results are sorted desc → rest are noise too
                cid = row["capture_id"]
                meta.setdefault(cid, {"title": row["title"] or "", "summary": row["summary"] or "",
                                      "created_at": _to_native(row["created_at"])})
                ids.append(cid)
        return ids

    async def _fulltext() -> list[str]:
        ids: list[str] = []
        async with get_session() as s:
            rec = await s.run(Q.FULLTEXT_SEARCH, search_text=q, limit=limit * 2)
            async for row in rec:
                cid = row["capture_id"]
                meta.setdefault(cid, {"title": row["title"] or "", "summary": row["summary"] or "",
                                      "created_at": _to_native(row["created_at"])})
                ids.append(cid)
        return ids

    tasks = {}
    if want_semantic and query_embedding:
        tasks["semantic"] = _semantic()
    if want_fulltext:
        tasks["fulltext"] = _fulltext()

    if tasks:
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for source, result in zip(tasks.keys(), done):
            if isinstance(result, Exception):
                logger.warning("Search source %s failed: %s", source, result)
                ranked[source] = []
            else:
                ranked[source] = result

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    fused: dict[str, float] = defaultdict(float)
    for source, ids in ranked.items():
        w = _RRF_WEIGHTS.get(source, 1.0)
        for rank, cid in enumerate(ids):
            fused[cid] += w / (_RRF_K + rank)

    # Normalize to 0..1 and drop the low-relevance tail. For a short query the
    # vector index returns *something* for every doc; RRF ranks them correctly
    # but the bottom is noise (e.g. 100/99/75% real matches, then 26-32% junk).
    # Keeping only results within 0.40 of the top yields a clean, precise list.
    top = max(fused.values()) if fused else 1.0
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    results = [
        SearchResult(
            capture_id=cid,
            title=meta[cid]["title"],
            summary=meta[cid]["summary"],
            score=round(score / top, 4),
            created_at=meta[cid]["created_at"],
        )
        for cid, score in ordered
        if (score / top) >= _MIN_RESULT_SCORE
    ][:limit]
    return SearchResponse(query=q, results=results, total=len(results))


# ── Re-extract topics from stored text (for captures with empty topics) ──────

@app.post("/reanalyze-topics/{capture_id}", summary="Re-extract topics using Claude (text-only)")
async def reanalyze_topics(capture_id: str):
    """
    Re-extracts topics, people, and places for a capture that has empty topics.
    Uses Claude text API on the stored summary + raw_text — no image needed.
    After updating topics, also rebuilds the embedding from the enriched text.
    """
    # Fetch existing capture
    async with get_session() as session:
        result = await session.run(Q.GET_CAPTURE_BY_ID, id=capture_id)
        record = await result.single()
        if record is None:
            raise HTTPException(status_code=404, detail=f"Capture '{capture_id}' not found")
        node = record["c"]

    title   = node.get("title", "") or ""
    summary = node.get("summary", "") or ""
    raw_text = node.get("raw_text", "") or ""

    if not summary and not raw_text:
        raise HTTPException(status_code=422, detail="No text content to analyze")

    # Call Claude to extract structured attributes from text
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = f"""Extract structured metadata from this screenshot content. Return ONLY valid JSON, no explanation.

Title: {title}
Summary: {summary}
Raw extracted text: {raw_text[:800]}

Return JSON with:
{{
  "category": "<exactly ONE value from the allowed list below>",
  "topics": ["<6-10 specific English topic keywords>"],
  "people": ["<real person names mentioned, or []>"],
  "places": ["<real place names or brands, or []>"]
}}

Allowed "category" values — pick the SINGLE best fit (coarse domain of the subject):
automotive, programming, electronics, computers, mobile, gaming, photography,
appliances, furniture, home, kitchen, tools, garden,
fashion, watches, jewelry, beauty, health, sports,
food, travel, music, entertainment, books, movies,
finance, education, office, toys, pets, baby

Category guidance (be CONSISTENT so similar items share a category):
- Use the real-world domain of the SUBJECT, not the fact that it's a web page/shop.
- ALL clothing, footwear, sportswear, apparel (shoes, sneakers, shorts, jackets) → "fashion"
- "sports" ONLY for sports equipment, fitness gear, sporting events/content (not clothing)
- Washing machine, oven, fridge, dishwasher, microwave → "appliances"
- Laptop/phone/TV/camera → "electronics"; sofa/table/bed → "furniture"
- Pick the closest match even if imperfect; this enables weak category-level links.

Rules for topics:
- Use canonical/standardized English terms so related captures share the same keywords
- If content is clearly about a known brand/product, include it (e.g. 'Tesla', 'YouTube', 'Steam', 'Java')
- Group synonyms: use 'electric vehicle' (not both 'electric car' AND 'electric vehicle')
- 'Premium Long Range', WLTP range, rear-wheel drive pricing → almost certainly Tesla configurator → add topic 'Tesla'
- Prefer specific over generic: 'Tesla Model 3' > 'car'; 'Java programming' > 'code'
- Do NOT put the category word itself into topics — it is returned separately"""

    try:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_json = next((b.text.strip() for b in resp.content if b.type == "text"), "{}")
        # Strip markdown code fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        import json as _json
        extracted = _json.loads(raw_json)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Claude extraction failed: {exc}") from exc

    new_topics  = [t for t in extracted.get("topics", []) if isinstance(t, str)][:15]
    new_people  = [p for p in extracted.get("people", []) if isinstance(p, str)][:10]
    new_places  = [pl for pl in extracted.get("places", []) if isinstance(pl, str)][:10]

    # Validate category against the controlled vocabulary; store as a Topic too
    from src.link_engine import DOMAIN_TOPICS
    category = (extracted.get("category") or "").strip().lower()
    if category and category in DOMAIN_TOPICS:
        # Add the domain tag as a topic so it participates in linking
        if category not in [t.lower() for t in new_topics]:
            new_topics.append(category)
    else:
        category = ""  # invalid / missing — skip

    # Persist new topics/people/places to Neo4j
    async with get_session() as session:
        # Remove old HAS_TOPIC / MENTIONS relationships before re-adding
        await session.run(
            "MATCH (c:Capture {id: $id})-[r:HAS_TOPIC|MENTIONS]->() DELETE r",
            id=capture_id,
        )
        for t in new_topics:
            await session.run(Q.UPSERT_TOPIC,  name=t,  capture_id=capture_id)
        for p in new_people:
            await session.run(Q.UPSERT_PERSON, name=p,  capture_id=capture_id)
        for pl in new_places:
            await session.run(Q.UPSERT_PLACE,  name=pl, capture_id=capture_id)

    # NOTE: Embedding is intentionally NOT recomputed here.
    # The Voyage free tier allows only 3 requests/minute, so re-analyzing many
    # captures in a row would hit the rate limit. Instead, call POST /reembed
    # (which batches ALL texts into a single Voyage request) afterwards.
    logger.info(
        "Reanalyzed %s: category=%s, topics=%s, people=%s, places=%s",
        capture_id[:8], category, new_topics, new_people, new_places,
    )

    return {
        "capture_id": capture_id,
        "category": category,
        "topics":  new_topics,
        "people":  new_people,
        "places":  new_places,
        "note": "Run POST /reembed to refresh embeddings (batched, avoids rate limit)",
    }


# ── Re-embed all captures (fixes zero embeddings from before Voyage AI key) ──

@app.post("/reembed", summary="Recompute embeddings for all captures")
async def reembed_all(force: bool = Query(default=False, description="Re-embed ALL, not just zero vectors")):
    """
    Recomputes Voyage AI embeddings for captures with zero/missing embeddings.

    Uses a SINGLE batched Voyage request to avoid the free-tier 3 RPM rate limit
    (embedding 12 captures one-by-one would otherwise fail after the first 3).

    Pass ?force=true to re-embed every capture (e.g. after changing embed text).
    """
    async with get_session() as session:
        result = await session.run(Q.GET_ALL_CAPTURES_FOR_REEMBED)
        records = [r async for r in result]

    # Collect captures that need (re)embedding
    to_embed: list[tuple[str, str]] = []  # (capture_id, embed_text)
    skipped = 0
    for rec in records:
        existing_emb = rec["embedding"] or []
        has_valid = existing_emb and any(v != 0.0 for v in existing_emb[:10])
        if has_valid and not force:
            skipped += 1
            continue
        title   = rec["title"] or ""
        summary = rec["summary"] or ""
        to_embed.append((rec["id"], f"{title} {summary}"))

    if not to_embed:
        return {
            "status": "done", "updated": 0, "skipped": skipped, "failed": 0,
            "message": f"Nic k přepočtu (přeskočeno {skipped}).",
        }

    # ONE batched Voyage request for all texts
    from src.vision_extractor import getEmbeddingsBatch
    embeddings = getEmbeddingsBatch([t for _, t in to_embed])

    updated = 0
    failed = 0
    async with get_session() as session:
        for (cap_id, _), emb in zip(to_embed, embeddings):
            if any(v != 0.0 for v in emb[:10]):
                await session.run(Q.UPDATE_EMBEDDING, id=cap_id, embedding=emb)
                updated += 1
            else:
                failed += 1

    return {
        "status": "done",
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "message": f"Přepočítáno: {updated}, přeskočeno (OK): {skipped}, chyby: {failed}",
    }


# ── Merge near-duplicate captures ────────────────────────────────────────────

import re as _re


def _norm_title(t: str) -> str:
    """Normalize a title for duplicate comparison: lowercase, alnum-only."""
    return _re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _title_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity of two normalized titles."""
    ta, tb = set(_norm_title(a).split()), set(_norm_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _are_same_entity(title_a: str, title_b: str, sim: float) -> bool:
    """
    Decide whether two captures are the SAME real-world entity (mergeable).

    Embedding similarity alone is not enough — a YouTube video about a Tesla and
    a Tesla article can both score ~0.95 yet are different captures. We require
    title agreement as well:
        • identical normalized title                          → merge
        • OR very high embedding (≥0.97) AND title Jaccard ≥0.6 → merge
    """
    if _norm_title(title_a) and _norm_title(title_a) == _norm_title(title_b):
        return True
    return sim >= 0.97 and _title_jaccard(title_a, title_b) >= 0.6


async def _merge_capture_nodes(session, keep_id: str, drop_id: str) -> None:
    """Move every edge from `drop` onto `keep`, record provenance, delete `drop`."""
    await session.run(Q.MERGE_MOVE_TOPICS,    keep=keep_id, drop=drop_id)
    await session.run(Q.MERGE_MOVE_MENTIONS,  keep=keep_id, drop=drop_id)
    await session.run(Q.MERGE_MOVE_LINKS_OUT, keep=keep_id, drop=drop_id)
    await session.run(Q.MERGE_MOVE_LINKS_IN,  keep=keep_id, drop=drop_id)
    await session.run(Q.MERGE_FINALIZE,       keep=keep_id, drop=drop_id)


@app.post("/merge-duplicates", summary="Merge near-duplicate captures into one node")
async def merge_duplicates(threshold: float = Query(default=0.88, ge=0.80, le=1.0)):
    """
    Finds captures that are the SAME real-world entity (e.g. two screenshots of
    the same product page) and merges each duplicate into the oldest canonical
    capture — transferring its topics, entities and links, then deleting it.

    `threshold` is the candidate gate: only pairs at least this similar are
    considered. The final merge decision additionally requires title agreement
    (identical normalized title, or ≥0.6 token overlap at ≥0.97 sim) so that
    merely-similar-but-distinct captures (e.g. a Tesla video vs a Tesla article,
    both ~0.95) are never merged. Self-links from the merge are cleaned up after.
    """
    async with get_session() as session:
        result = await session.run(Q.GET_ALL_CAPTURES_FULL)
        records = [r async for r in result]

    # Build lookup; iterate oldest-first so the canonical is the earliest capture
    by_id = {r["id"]: {"title": r["title"] or "", "embedding": list(r["embedding"] or [])}
             for r in records}
    order = [r["id"] for r in records]
    removed: set[str] = set()
    merges: list[dict] = []

    async with get_session() as session:
        for cid in order:
            if cid in removed:
                continue
            emb = by_id[cid]["embedding"]
            if not emb or not any(v != 0.0 for v in emb[:10]):
                continue

            # Find vector neighbours of this capture
            try:
                rec = await session.run(
                    Q.SEMANTIC_SEARCH, embedding=emb, k=10, limit=10
                )
                neighbours = [r async for r in rec]
            except Exception as exc:
                logger.warning("Duplicate scan failed for %s: %s", cid[:8], exc)
                continue

            for n in neighbours:
                nid = n["capture_id"]
                if nid == cid or nid in removed:
                    continue
                sim = float(n["score"])
                if sim < threshold:
                    continue  # results are sorted desc → no closer ones remain
                if _are_same_entity(by_id[cid]["title"], by_id.get(nid, {}).get("title", n["title"] or ""), sim):
                    await _merge_capture_nodes(session, keep_id=cid, drop_id=nid)
                    removed.add(nid)
                    merges.append({
                        "kept": cid, "kept_title": by_id[cid]["title"],
                        "merged": nid, "similarity": round(sim, 3),
                    })
                    logger.info("Merged duplicate %s → %s (sim=%.3f)", nid[:8], cid[:8], sim)

        # Remove any accidental self-links left behind
        await session.run("MATCH (c:Capture)-[r:LINKED_TO]->(c) DELETE r")

    return {
        "status": "done",
        "merged_pairs": len(merges),
        "remaining_captures": len(order) - len(removed),
        "details": merges,
        "message": f"Sloučeno {len(merges)} duplicit, zůstává {len(order) - len(removed)} záznamů.",
    }


# ── Subject branches: group same-entity captures across sources ──────────────

_DOMAIN_RE = _re.compile(
    r"\b([a-z0-9][a-z0-9\-]{1,}\.(?:cz|com|sk|net|org|io|eu|de|shop|store))\b", _re.I
)


def _extract_source(title: str, summary: str, raw_text: str) -> str:
    """Best-effort source label for a capture (website domain, else generic)."""
    for txt in (title, summary, raw_text):
        m = _DOMAIN_RE.search(txt or "")
        if m:
            return m.group(1).lower()
    return "neznámý zdroj"


def _distinctive_tokens(title: str) -> set[str]:
    """Model-number-like tokens (letters+digits) — strong same-product anchors."""
    return {
        t for t in _norm_title(title).split()
        if len(t) >= 5 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    }


def _same_subject(title_a: str, title_b: str, sim: float) -> bool:
    """
    Looser than _are_same_entity (merge): grouping is non-destructive, so we can
    be more permissive. Two captures depict the same Subject if they are quite
    similar AND share strong title evidence (identical title, a shared model
    number, or solid token overlap).
    """
    if sim < 0.85:
        return False
    if _norm_title(title_a) and _norm_title(title_a) == _norm_title(title_b):
        return True
    if _distinctive_tokens(title_a) & _distinctive_tokens(title_b):
        return True
    return _title_jaccard(title_a, title_b) >= 0.5


def _clean_subject_title(title: str) -> str:
    """Strip source suffix / boilerplate to get a clean canonical Subject name."""
    s = title or ""
    # remove " on Alza.cz", " - mall.cz", " | shop.com" …
    s = _re.sub(
        r"\s*(?:on|@|[-|–—])\s*[A-Za-z0-9.\- ]*\.(?:cz|com|sk|net|org|io|eu|de|shop|store)\b.*$",
        "", s, flags=_re.I,
    )
    # remove trailing boilerplate words
    s = _re.sub(r"\s*[-|–—]?\s*\b(?:product page|product listing|listing|product|page)\b\s*$",
                "", s, flags=_re.I)
    return s.strip(" -|–—") or (title or "").strip()


@app.post("/group-subjects", summary="Group captures of the same thing into Subject branches")
async def group_subjects(threshold: float = Query(default=0.85, ge=0.70, le=1.0)):
    """
    Builds :Subject branches. Captures depicting the same real-world thing — the
    same product seen on Alza / another e-shop / a personal photo — are grouped
    under one (Capture)-[:DEPICTS]->(Subject) hub, WITHOUT deleting any capture.
    Each capture keeps its `source`, so the branch shows where you saw the thing.

    Non-destructive: re-runnable. Rebuilds all Subjects from scratch each call.
    """
    import uuid

    async with get_session() as session:
        result = await session.run(Q.GET_ALL_CAPTURES_FULL)
        records = [r async for r in result]

    caps = {
        r["id"]: {
            "title": r["title"] or "",
            "embedding": list(r["embedding"] or []),
            "source": _extract_source(r["title"] or "", r["summary"] or "", r["raw_text"] or ""),
        }
        for r in records
    }
    order = [r["id"] for r in records]

    # Union-find to cluster same-subject captures
    parent: dict[str, str] = {cid: cid for cid in order}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    async with get_session() as session:
        # Persist each capture's source label
        for cid, c in caps.items():
            await session.run(Q.SET_CAPTURE_SOURCE, id=cid, source=c["source"])

        # Discover same-subject pairs via the vector index
        for cid in order:
            emb = caps[cid]["embedding"]
            if not emb or not any(v != 0.0 for v in emb[:10]):
                continue
            try:
                rec = await session.run(Q.SEMANTIC_SEARCH, embedding=emb, k=10, limit=10)
                neighbours = [r async for r in rec]
            except Exception as exc:
                logger.warning("Subject scan failed for %s: %s", cid[:8], exc)
                continue
            for n in neighbours:
                nid = n["capture_id"]
                if nid == cid or nid not in caps:
                    continue
                sim = float(n["score"])
                if sim < threshold:
                    continue
                if _same_subject(caps[cid]["title"], caps[nid]["title"], sim):
                    union(cid, nid)

        # Build clusters
        clusters: dict[str, list[str]] = {}
        for cid in order:
            clusters.setdefault(find(cid), []).append(cid)

        # Rebuild Subjects from scratch
        await session.run(Q.CLEAR_SUBJECTS)
        subjects = []
        for members in clusters.values():
            if len(members) < 2:
                continue  # a lone capture needs no Subject branch
            titles = [caps[m]["title"] for m in members]
            # canonical name = most common cleaned title (tie → shortest)
            cleaned = [_clean_subject_title(t) for t in titles]
            name = min(
                sorted(set(cleaned), key=cleaned.count, reverse=True)[:1] or cleaned,
                key=len,
            )
            sources = sorted({caps[m]["source"] for m in members})
            kind = "produkt" if any("." in s for s in sources) else "věc"
            sid = str(uuid.uuid4())
            await session.run(
                Q.UPSERT_SUBJECT, id=sid, name=name, kind=kind, capture_ids=members
            )
            subjects.append({"name": name, "kind": kind, "sources": sources, "captures": len(members)})

    return {
        "status": "done",
        "subjects": len(subjects),
        "details": subjects,
        "message": (
            f"Vytvořeno {len(subjects)} větví (předmětů). "
            + "; ".join(f"{s['name']} ({s['captures']}× — {', '.join(s['sources'])})" for s in subjects)
            if subjects else "Žádné skupiny stejných předmětů nenalezeny."
        ),
    }


# ── Relink all captures (recompute LINKED_TO with current engine settings) ───

@app.post("/relink", summary="Delete all links and recompute them from scratch")
async def relink_all():
    """
    Deletes every LINKED_TO relationship and recomputes links
    using the current engine settings (thresholds, priority rules, etc.).

    Use this after:
    - Changing link strength thresholds
    - Uploading new captures (to backfill links to old captures)
    - Any change to link_engine.py logic
    """
    # Step 1: Delete all existing links
    async with get_session() as session:
        await session.run(Q.DELETE_ALL_LINKS)
    logger.info("Deleted all existing LINKED_TO relationships")

    # Step 2: Load all captures with their embeddings
    async with get_session() as session:
        result = await session.run(Q.GET_ALL_CAPTURES_FOR_RELINK)
        records = [r async for r in result]

    total_captures = len(records)
    total_links = 0
    failed = 0

    # Step 3: For each capture, find similar and persist links
    for rec in records:
        cap_id = rec["id"]
        embedding = list(rec["embedding"] or [])

        # Skip captures without a valid embedding
        if not embedding or not any(v != 0.0 for v in embedding[:10]):
            logger.warning("Skipping capture %s — no valid embedding", cap_id[:8])
            continue

        try:
            linked = await findSimilarCaptures(
                cap_id,
                ExtractedAttributes(),  # topics/people come from Neo4j directly
                embedding,
            )
            if linked:
                await persistLinks(cap_id, linked)
                total_links += len(linked)
                logger.info("Relinked %s → %d links", cap_id[:8], len(linked))
        except Exception as exc:
            logger.error("Relink failed for %s: %s", cap_id[:8], exc)
            failed += 1

    return {
        "status": "done",
        "captures_processed": total_captures,
        "links_created": total_links,
        "failed": failed,
        "message": (
            f"Zpracováno {total_captures} záznamů, "
            f"vytvořeno {total_links} vazeb, "
            f"chyby: {failed}"
        ),
    }


# ── Graph data (for web UI visualization) ────────────────────────────────────

@app.get("/graph-data", summary="All nodes + edges for graph visualization")
async def get_graph_data():
    """Returns all Capture nodes and LINKED_TO edges in vis-network format."""
    nodes = []
    edges = []

    async with get_session() as session:
        # All captures
        result = await session.run(
            "MATCH (c:Capture) "
            "RETURN c.id AS id, c.title AS title, c.summary AS summary, "
            "       c.created_at AS created_at, coalesce(c.merged_count, 1) AS merged_count"
        )
        async for rec in result:
            nodes.append({
                "id": rec["id"],
                "label": (rec["title"] or "")[:35],
                "title": (rec["summary"] or "")[:120],
                "created_at": str(rec["created_at"]),
                "merged_count": rec["merged_count"],
                "type": "capture",
            })

        # All links
        result = await session.run(
            "MATCH (a:Capture)-[r:LINKED_TO]->(b:Capture) "
            "RETURN a.id AS from_id, b.id AS to_id, "
            "       r.strength AS strength, r.type AS link_type"
        )
        async for rec in result:
            edges.append({
                "from": rec["from_id"],
                "to": rec["to_id"],
                "value": float(rec["strength"] or 0.5),
                "title": rec["link_type"] or "",
                "type": "link",
            })

        # Subject branches: a hub node per Subject + a DEPICTS branch per capture
        result = await session.run(Q.GET_SUBJECTS_GRAPH)
        async for rec in result:
            sid = rec["subject_id"]
            nodes.append({
                "id": sid,
                "label": rec["subject_name"] or "Předmět",
                "title": f"{rec['subject_kind'] or 'předmět'} · {len(rec['capture_ids'])} zdrojů",
                "type": "subject",
            })
            for cid in rec["capture_ids"]:
                edges.append({
                    "from": sid,
                    "to": cid,
                    "value": 1.0,
                    "title": "DEPICTS",
                    "type": "depicts",
                })

    return {"nodes": nodes, "edges": edges}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "env": settings.app_env}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.app_log_level,
    )
