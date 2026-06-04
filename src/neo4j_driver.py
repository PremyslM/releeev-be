"""
Neo4j connection management — singleton driver with health-check.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import settings

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    """Return (and lazily create) the global Neo4j async driver."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
        await _verify_connectivity(_driver)
        logger.info("Neo4j driver initialised — %s", settings.neo4j_uri)
    return _driver


async def _verify_connectivity(driver: AsyncDriver) -> None:
    try:
        await driver.verify_connectivity()
    except ServiceUnavailable as exc:
        logger.error("Neo4j unavailable at %s: %s", settings.neo4j_uri, exc)
        raise
    except AuthError as exc:
        logger.error("Neo4j auth failed: %s", exc)
        raise


async def close_driver() -> None:
    """Close the global driver — call on app shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed.")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a Neo4j session."""
    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        yield session
