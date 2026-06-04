"""
Brain Graph — FastAPI application.

Endpoints:
    POST /analyze-capture
    GET  /captures/{id}
    GET  /search?q=...
"""
from __future__ import annotations

import hashlib
import logging
import time
from contextlib import asynccontextmanager

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

        # Parse attributes from flat node properties
        attrs = ExtractedAttributes(
            title=node.get("title", ""),
            summary=node.get("summary", ""),
            raw_text=node.get("raw_text", ""),
            language=node.get("language", "en"),
            sentiment=node.get("sentiment", "neutral"),
            confidence=node.get("confidence", 1.0),
        )
        capture = CaptureNode(
            id=node["id"],
            image_hash=node.get("image_hash", ""),
            attributes=attrs,
            embedding=list(node.get("embedding") or []),
        )

        # Fetch connections
        conn_result = await session.run(Q.GET_CAPTURE_CONNECTIONS, id=capture_id)
        connections: list[CaptureLink] = []
        async for row in conn_result:
            rel = row["r"]
            other = row["other"]
            try:
                lt = LinkType(rel.get("type", "TOPIC_OVERLAP"))
            except ValueError:
                lt = LinkType.topic_overlap

            connections.append(
                CaptureLink(
                    source_id=capture_id,
                    target_id=other["id"],
                    link_type=lt,
                    strength=float(rel.get("strength", 0.0)),
                )
            )

    return CaptureDetailResponse(capture=capture, connections=connections)


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
        # Combine original + expanded terms
        combined = f"{q} {expanded_text.replace(',', ' ')}"
        logger.info("Query expanded: %r → %r", q, combined[:100])
        return combined
    except Exception as exc:
        logger.warning("Query expansion failed (using original): %s", exc)
        return q


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic + full-text search across captures (multilingual)",
)
async def search_captures(
    q: str = Query(..., min_length=1, description="Search query (any language)"),
    limit: int = Query(default=20, ge=1, le=100),
    mode: str = Query(default="hybrid", description="'hybrid' | 'semantic' | 'fulltext'"),
    expand: bool = Query(default=True, description="Use Claude to expand/translate query"),
) -> SearchResponse:
    """
    Multilingual search with Claude query expansion:
    - Expands short queries (e.g. 'hra' → 'game gaming video game play')
    - Combines semantic embedding search + fulltext
    - Boosts captures that have exact Topic matches
    """
    seen: dict[str, SearchResult] = {}  # capture_id → best result

    # ── Query expansion via Claude ────────────────────────────────────────────
    effective_query = _expand_query_with_claude(q) if expand else q

    # ── Generate query embedding (for semantic / hybrid) ──────────────────────
    query_embedding: list[float] = []
    if mode in ("semantic", "hybrid"):
        try:
            from src.vision_extractor import _voyage
            vc = _voyage()
            result = vc.embed(texts=[effective_query], model="voyage-3", input_type="query")
            query_embedding = result.embeddings[0]
        except Exception as exc:
            logger.warning("Embedding for search query failed: %s", exc)

    async with get_session() as session:

        # ── Topic-based boost (exact / partial keyword match) ─────────────────
        # Run for each word in the original query
        for keyword in q.split():
            if len(keyword) < 3:
                continue
            try:
                rec = await session.run(
                    Q.TOPIC_BOOST_SEARCH,
                    keyword=keyword,
                    limit=limit,
                )
                async for row in rec:
                    cid = row["capture_id"]
                    created_at = row["created_at"]
                    if hasattr(created_at, "to_native"):
                        created_at = created_at.to_native()
                    boost_score = float(row["score"])  # 1.5 — topic exact match
                    if cid not in seen or boost_score > seen[cid].score:
                        seen[cid] = SearchResult(
                            capture_id=cid,
                            title=row["title"] or "",
                            summary=row["summary"] or "",
                            score=boost_score,
                            created_at=created_at,
                        )
            except Exception as exc:
                logger.debug("Topic boost search failed: %s", exc)

        # ── Semantic search ───────────────────────────────────────────────────
        if mode in ("semantic", "hybrid") and query_embedding:
            try:
                rec = await session.run(
                    Q.SEMANTIC_SEARCH,
                    embedding=query_embedding,
                    k=limit * 2,
                    limit=limit,
                )
                async for row in rec:
                    cid = row["capture_id"]
                    created_at = row["created_at"]
                    if hasattr(created_at, "to_native"):
                        created_at = created_at.to_native()
                    sem_score = float(row["score"])
                    # Add to topic-boosted score if already found, else set
                    if cid in seen:
                        seen[cid] = SearchResult(
                            capture_id=cid,
                            title=seen[cid].title,
                            summary=seen[cid].summary,
                            score=seen[cid].score + sem_score,  # combine
                            created_at=seen[cid].created_at,
                        )
                    else:
                        seen[cid] = SearchResult(
                            capture_id=cid,
                            title=row["title"] or "",
                            summary=row["summary"] or "",
                            score=sem_score,
                            created_at=created_at,
                        )
            except Exception as exc:
                logger.warning("Semantic search failed (non-fatal): %s", exc)

        # ── Fulltext search ───────────────────────────────────────────────────
        if mode in ("fulltext", "hybrid"):
            try:
                rec = await session.run(
                    Q.FULLTEXT_SEARCH,
                    search_text=q,
                    limit=limit,
                )
                async for row in rec:
                    cid = row["capture_id"]
                    created_at = row["created_at"]
                    if hasattr(created_at, "to_native"):
                        created_at = created_at.to_native()
                    ft_score = float(row["score"]) * 0.8
                    if cid in seen:
                        seen[cid] = SearchResult(
                            capture_id=cid,
                            title=seen[cid].title,
                            summary=seen[cid].summary,
                            score=seen[cid].score + ft_score,  # combine
                            created_at=seen[cid].created_at,
                        )
                    else:
                        seen[cid] = SearchResult(
                            capture_id=cid,
                            title=row["title"] or "",
                            summary=row["summary"] or "",
                            score=ft_score,
                            created_at=created_at,
                        )
            except Exception as exc:
                logger.warning("Fulltext search failed (non-fatal): %s", exc)

    # Sort by score descending, return top `limit`
    results = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:limit]

    return SearchResponse(query=q, results=results, total=len(results))


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
