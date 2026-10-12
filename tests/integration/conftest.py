"""Integration test fixtures (shared DB such as spark.home / app_db)."""

from __future__ import annotations

import asyncio
import os

import pytest

from retrieval_config import load_retrieval_config
from retrieval_db import discard_pool, run_migrations


def database_url() -> str | None:
    url = os.getenv("RETRIEVAL_DATABASE_URL", "").strip()
    return url or None


@pytest.fixture(scope="session", autouse=True)
def _integration_run_migrations_once() -> None:
    if database_url() is None:
        return

    async def _migrate() -> None:
        await run_migrations(load_retrieval_config())
        discard_pool()

    asyncio.run(_migrate())


@pytest.fixture(autouse=True)
def _integration_discard_pool_after_test(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("integration") is None:
        yield
        return
    yield
    if database_url() is not None:
        discard_pool()
