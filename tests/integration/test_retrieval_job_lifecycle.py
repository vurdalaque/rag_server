"""Integration tests for evaluation job state transitions and exhausted-attempt cleanup."""

from __future__ import annotations

import asyncio

import pytest

from tests.integration.conftest import database_url
from tests.integration.helpers import seed_event_and_running_job

pytestmark = pytest.mark.integration
pytest.importorskip("asyncpg")

from retrieval_config import load_retrieval_config
from retrieval_db import (
    claim_evaluation_jobs,
    commit_evaluation_and_complete_job,
    get_pool,
    mark_exhausted_evaluation_jobs,
)


def test_active_last_attempt_not_marked_dead() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        _event_id, _created_at, job_id, _token, evaluation_version = await seed_event_and_running_job(
            prefix="active-last",
            max_attempts=1,
            attempts=1,
        )
        marked = await mark_exhausted_evaluation_jobs(evaluation_version)
        assert marked == 0
        pool = await get_pool(load_retrieval_config())
        status = await pool.fetchval(
            "SELECT status FROM retrieval_obs.evaluation_jobs WHERE job_id = $1",
            job_id,
        )
        assert status == "running"

    asyncio.run(_run())


def test_expired_lease_last_attempt_marked_dead() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        _event_id, _created_at, job_id, _token, evaluation_version = await seed_event_and_running_job(
            prefix="expired-last",
            max_attempts=1,
            attempts=1,
        )
        pool = await get_pool(load_retrieval_config())
        await pool.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET lease_expires_at = now() - interval '1 minute'
            WHERE job_id = $1
            """,
            job_id,
        )
        marked = await mark_exhausted_evaluation_jobs(evaluation_version)
        assert marked == 1
        status = await pool.fetchval(
            "SELECT status FROM retrieval_obs.evaluation_jobs WHERE job_id = $1",
            job_id,
        )
        assert status == "dead"
        reclaimed = await claim_evaluation_jobs(evaluation_version, 5, 60, "worker-b")
        assert all(int(j["job_id"]) != job_id for j in reclaimed)

    asyncio.run(_run())


def test_claim_cleanup_does_not_kill_active_last_attempt() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, token, evaluation_version = await seed_event_and_running_job(
            prefix="claim-cleanup",
            max_attempts=2,
            attempts=2,
        )
        pool = await get_pool(load_retrieval_config())
        await claim_evaluation_jobs(evaluation_version, 1, 60, "worker-b")
        status = await pool.fetchval(
            "SELECT status FROM retrieval_obs.evaluation_jobs WHERE job_id = $1",
            job_id,
        )
        assert status == "running"
        ok = await commit_evaluation_and_complete_job(
            job_id,
            token,
            {
                "event_id": event_id,
                "event_created_at": created_at,
                "evaluation_version": evaluation_version,
                "judge_model": "judge-a",
                "prompt_version": "v1",
                "per_chunk_scores": [],
                "context_sufficiency": {},
                "query_classification": {},
                "aggregated_metrics": {},
                "explanation": "ok",
                "status": "completed",
                "diagnostics": {},
                "truncation_info": {},
            },
        )
        assert ok is True

    asyncio.run(_run())


def test_migration_011_duplicate_guard() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        pool = await get_pool(load_retrieval_config())
        dup_groups = await pool.fetchval(
            """
            SELECT COUNT(*)::int FROM (
                SELECT 1
                FROM retrieval_obs.retrieval_evaluations
                WHERE status = 'completed'
                GROUP BY event_id, event_created_at, evaluation_version
                HAVING COUNT(*) > 1
            ) t
            """
        )
        assert dup_groups == 0

    asyncio.run(_run())
