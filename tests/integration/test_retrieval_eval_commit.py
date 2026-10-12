"""Integration tests for evaluation commit lease fencing and evaluation identity."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from tests.integration.conftest import database_url
from tests.integration.helpers import insert_test_event, new_evaluation_version, seed_event_and_running_job

pytestmark = pytest.mark.integration
pytest.importorskip("asyncpg")

from retrieval_config import load_retrieval_config
from retrieval_db import (
    claim_evaluation_jobs,
    commit_evaluation_and_complete_job,
    enqueue_evaluation_job,
    get_pool,
    reconcile_missing_evaluation_jobs,
)


def _evaluation_row(
    *,
    event_id: uuid.UUID,
    created_at: datetime,
    evaluation_version: str,
    judge_model: str,
    prompt_version: str,
) -> dict:
    return {
        "event_id": event_id,
        "event_created_at": created_at,
        "evaluation_version": evaluation_version,
        "judge_model": judge_model,
        "prompt_version": prompt_version,
        "per_chunk_scores": [{"chunk_index": 0, "relevance": 2}],
        "context_sufficiency": {"sufficient": True, "missing_topics": []},
        "query_classification": {"category": "answerable"},
        "aggregated_metrics": {"precision_at_k": 1.0},
        "explanation": "ok",
        "status": "completed",
        "diagnostics": {},
        "truncation_info": {},
    }


def test_stale_worker_cannot_commit_after_lease_reclaim() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, stale_token, evaluation_version = await seed_event_and_running_job(
            prefix="stale-lease",
        )
        config = load_retrieval_config()
        pool = await get_pool(config)
        async with pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE retrieval_obs.evaluation_jobs
                SET lease_expires_at = now() - interval '1 minute'
                WHERE job_id = $1
                """,
                job_id,
            )

        new_jobs = await claim_evaluation_jobs(evaluation_version, 1, 60, "w2")
        assert len(new_jobs) == 1
        assert new_jobs[0]["job_id"] == job_id or new_jobs[0]["event_id"] == event_id

        ok = await commit_evaluation_and_complete_job(
            job_id,
            stale_token,
            _evaluation_row(
                event_id=event_id,
                created_at=created_at,
                evaluation_version=evaluation_version,
                judge_model="judge-a",
                prompt_version="v1",
            ),
        )
        assert ok is False

        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = $2
            """,
            event_id,
            evaluation_version,
        )
        assert count == 0

    asyncio.run(_run())


def test_commit_is_idempotent_for_same_evaluation_version() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, token, evaluation_version = await seed_event_and_running_job(
            prefix="idempotent",
        )
        row = _evaluation_row(
            event_id=event_id,
            created_at=created_at,
            evaluation_version=evaluation_version,
            judge_model="judge-a",
            prompt_version="v1",
        )
        assert await commit_evaluation_and_complete_job(job_id, token, row) is True

        pool = await get_pool(load_retrieval_config())
        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = $2 AND status = 'completed'
            """,
            event_id,
            evaluation_version,
        )
        assert count == 1

    asyncio.run(_run())


def test_different_judge_models_same_prompt_create_distinct_evaluations() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at = await insert_test_event("dual-model")
        config = load_retrieval_config()
        pool = await get_pool(config)

        for model in ("model-a", "model-b"):
            version = new_evaluation_version(f"dual-{model}")
            assert await enqueue_evaluation_job(event_id, created_at, version, max_attempts=3)
            jobs = await claim_evaluation_jobs(version, 1, 60, "worker")
            assert len(jobs) == 1
            job = jobs[0]
            assert job["event_id"] == event_id
            async with pool.acquire() as connection:
                await connection.execute(
                    """
                    UPDATE retrieval_obs.evaluation_jobs
                    SET status = 'running', lease_expires_at = now() + interval '5 minutes'
                    WHERE job_id = $1
                    """,
                    job["job_id"],
                )
            assert await commit_evaluation_and_complete_job(
                int(job["job_id"]),
                job["lease_token"],
                _evaluation_row(
                    event_id=event_id,
                    created_at=created_at,
                    evaluation_version=version,
                    judge_model=model,
                    prompt_version="v1",
                ),
            )

        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND prompt_version = 'v1' AND status = 'completed'
            """,
            event_id,
        )
        assert count == 2

    asyncio.run(_run())


def test_reconciliation_respects_evaluation_version() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at = await insert_test_event("reconcile")
        evaluation_version = new_evaluation_version("reconcile")
        inserted = await reconcile_missing_evaluation_jobs(
            evaluation_version,
            limit=10,
            max_attempts=3,
            event_id=event_id,
        )
        assert inserted == 1
        inserted_again = await reconcile_missing_evaluation_jobs(
            evaluation_version,
            limit=10,
            max_attempts=3,
            event_id=event_id,
        )
        assert inserted_again == 0

    asyncio.run(_run())


def test_migrations_preserve_existing_evaluations() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        pool = await get_pool(load_retrieval_config())
        event_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        eval_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO retrieval_obs.retrieval_evaluations (
                evaluation_id, event_id, event_created_at,
                evaluation_version, judge_model, prompt_version,
                per_chunk_scores, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, '[]'::jsonb, 'completed')
            """,
            eval_id,
            event_id,
            created_at,
            "legacy:v1:legacy-model",
            "legacy-model",
            "legacy:v1",
        )
        row = await pool.fetchrow(
            """
            SELECT evaluation_version, judge_model, prompt_version
            FROM retrieval_obs.retrieval_evaluations
            WHERE evaluation_id = $1
            """,
            eval_id,
        )
        assert row is not None
        assert row["evaluation_version"] == "legacy:v1:legacy-model"

    asyncio.run(_run())


def test_concurrent_commits_only_one_succeeds() -> None:
    if database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, token, evaluation_version = await seed_event_and_running_job(
            prefix="concurrent",
        )
        row = _evaluation_row(
            event_id=event_id,
            created_at=created_at,
            evaluation_version=evaluation_version,
            judge_model="judge-a",
            prompt_version="v1",
        )

        async def attempt() -> bool:
            return await commit_evaluation_and_complete_job(job_id, token, row)

        results = await asyncio.gather(attempt(), attempt())
        assert sum(1 for ok in results if ok) == 1

        pool = await get_pool(load_retrieval_config())
        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = $2
            """,
            event_id,
            evaluation_version,
        )
        assert count == 1

    asyncio.run(_run())
