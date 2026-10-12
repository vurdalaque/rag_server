"""Integration tests against a live TimescaleDB/PostgreSQL (optional).

Run:
  docker compose -f docker-compose.integration.yml up -d
  RETRIEVAL_DATABASE_URL=postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval \\
    pytest tests/integration/test_retrieval_db.py -m integration -v
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest

from retrieval_config import load_retrieval_config
from retrieval_db import close_pool, get_pool, insert_events, run_migrations

pytestmark = pytest.mark.integration

pytest.importorskip("asyncpg")


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


def test_insert_event_roundtrip() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        event_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        inserted = await insert_events(
            [
                {
                    "event_id": event_id,
                    "created_at": created_at,
                    "correlation_id": "integration-test",
                    "request_source": "pytest",
                    "status": "ok",
                    "request_payload": {"query": "hello"},
                    "response_payload": {"results": []},
                    "retrieval_trace": {},
                    "metadata": {},
                }
            ]
        )
        assert inserted == 1
        pool = await get_pool(config)
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT event_id, request_source
                FROM retrieval_obs.retrieval_events
                WHERE event_id = $1 AND created_at = $2
                """,
                event_id,
                created_at,
            )
        assert row is not None
        assert row["request_source"] == "pytest"
        await close_pool()

    asyncio.run(_run())
