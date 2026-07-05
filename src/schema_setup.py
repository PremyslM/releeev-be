"""
One-time schema bootstrap: constraints + indexes for all node types.

Run directly:  python -m src.schema_setup
Or called automatically during app startup.
"""
from __future__ import annotations

import asyncio
import logging

from src.neo4j_driver import get_session

logger = logging.getLogger(__name__)

# ── Constraints ──────────────────────────────────────────────────────────────
CONSTRAINTS = [
    # Each node type has a unique id
    "CREATE CONSTRAINT capture_id IF NOT EXISTS FOR (n:Capture) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (n:Topic)   REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (n:Person)  REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT place_name  IF NOT EXISTS FOR (n:Place)   REQUIRE n.name IS UNIQUE",
    # A Subject groups captures of the same real-world thing (product/person/…)
    "CREATE CONSTRAINT subject_id  IF NOT EXISTS FOR (n:Subject) REQUIRE n.id IS UNIQUE",
    # Users (auth)
    "CREATE CONSTRAINT user_id     IF NOT EXISTS FOR (n:User)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT user_email  IF NOT EXISTS FOR (n:User)    REQUIRE n.email IS UNIQUE",
]

# ── Indexes ───────────────────────────────────────────────────────────────────
INDEXES = [
    # Full-text search on Capture content
    """CREATE FULLTEXT INDEX capture_fulltext IF NOT EXISTS
       FOR (n:Capture) ON EACH [n.title, n.summary, n.raw_text]""",

    # Range index on timestamp for temporal queries
    "CREATE INDEX capture_created IF NOT EXISTS FOR (n:Capture) ON (n.created_at)",

    # Vector index for embedding similarity  (Neo4j 5.x+)
    # voyage-3 returns 1024-dimensional vectors
    """CREATE VECTOR INDEX capture_embedding IF NOT EXISTS
       FOR (n:Capture) ON n.embedding
       OPTIONS {indexConfig: {
           `vector.dimensions`: 1024,
           `vector.similarity_function`: 'cosine'
       }}""",
]


async def run_schema_setup() -> None:
    async with get_session() as session:
        # Drop vector index if it exists with wrong dimensions (1536 instead of 1024)
        try:
            result = await session.run(
                "SHOW INDEXES YIELD name, type, options WHERE name = 'capture_embedding'"
            )
            record = await result.single()
            if record:
                opts = record.get("options") or {}
                index_config = opts.get("indexConfig") or {}
                dims = index_config.get("vector.dimensions")
                if dims and int(dims) != 1024:
                    logger.info("Dropping vector index with wrong dimensions (%s)…", dims)
                    await session.run("DROP INDEX capture_embedding IF EXISTS")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not check/drop vector index: %s", exc)

        for stmt in CONSTRAINTS + INDEXES:
            try:
                await session.run(stmt)
                logger.info("Schema OK: %s…", stmt[:60])
            except Exception as exc:  # noqa: BLE001
                # Some Neo4j editions don't support vector indexes — skip gracefully
                logger.warning("Schema stmt skipped (%s): %s…", exc, stmt[:60])
    logger.info("Schema setup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_schema_setup())
