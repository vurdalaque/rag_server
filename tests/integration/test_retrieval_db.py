"""Integration tests against a live TimescaleDB/PostgreSQL (optional).

Run:
  RETRIEVAL_DATABASE_URL=postgresql://... pytest -m integration tests/integration/
"""

from __future__ import annotations

import asyncio
import os

import pytest

from retrieval_config import load_retrieval_config
from retrieval_db import close_pool, get_pool, run_migrations

pytestmark = pytest.mark.integration


def _database_url() -> str | None:
    url = os.getenv("RETRIEVAL_DATABASE_URL", "").strip()
    return url or None


def test_migrations_idempotent() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        second = await run_migrations(config)
        assert second == []
        await close_pool()

    asyncio.run(_run())


def test_storage_stats_view() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        pool = await get_pool(config)
        rows = await pool.fetch(
            "SELECT relation_name FROM retrieval_obs.table_byte_sizes LIMIT 5"
        )
        names = {row["relation_name"] for row in rows}
        assert "retrieval_events" in names
        await close_pool()

    asyncio.run(_run())
