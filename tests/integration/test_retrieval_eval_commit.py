"""Integration tests for evaluation commit lease fencing and evaluation identity.

Requires PostgreSQL/TimescaleDB:
  RETRIEVAL_DATABASE_URL=postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval \\
    pytest tests/integration/test_retrieval_eval_commit.py -m integration -v
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("asyncpg")

from retrieval_config import load_retrieval_config
from retrieval_db import (
    claim_evaluation_jobs,
    close_pool,
    commit_evaluation_and_complete_job,
    enqueue_evaluation_job,
    get_pool,
    insert_events,
    reconcile_missing_evaluation_jobs,
    run_migrations,
)


def _database_url() -> str | None:
    url = os.getenv("RETRIEVAL_DATABASE_URL", "").strip()
    return url or None


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


async def _seed_event_and_job(
    evaluation_version: str = "v1:judge-a",
) -> tuple[uuid.UUID, datetime, int, uuid.UUID]:
    config = load_retrieval_config()
    await run_migrations(config)
    event_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    await insert_events(
        [
            {
                "event_id": event_id,
                "created_at": created_at,
                "correlation_id": "commit-test",
                "request_source": "pytest",
                "status": "ok",
                "request_payload": {"query": "q"},
                "response_payload": {"results": []},
                "retrieval_trace": {},
                "metadata": {},
            }
        ]
    )
    assert await enqueue_evaluation_job(
        event_id,
        created_at,
        evaluation_version,
        max_attempts=3,
    )
    jobs = await claim_evaluation_jobs(evaluation_version, 1, 60, "w1")
    assert len(jobs) == 1
    job_id = int(jobs[0]["job_id"])
    token = jobs[0]["lease_token"]
    pool = await get_pool(config)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET status = 'running', lease_expires_at = now() + interval '5 minutes'
            WHERE job_id = $1
            """,
            job_id,
        )
    return event_id, created_at, job_id, token


def test_stale_worker_cannot_commit_after_lease_reclaim() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, stale_token = await _seed_event_and_job()
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

        new_jobs = await claim_evaluation_jobs("v1:judge-a", 1, 60, "w2")
        assert len(new_jobs) == 1

        ok = await commit_evaluation_and_complete_job(
            job_id,
            stale_token,
            _evaluation_row(
                event_id=event_id,
                created_at=created_at,
                evaluation_version="v1:judge-a",
                judge_model="judge-a",
                prompt_version="v1",
            ),
        )
        assert ok is False

        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = 'v1:judge-a'
            """,
            event_id,
        )
        assert count == 0
        await close_pool()

    asyncio.run(_run())


def test_commit_is_idempotent_for_same_evaluation_version() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, token = await _seed_event_and_job()
        row = _evaluation_row(
            event_id=event_id,
            created_at=created_at,
            evaluation_version="v1:judge-a",
            judge_model="judge-a",
            prompt_version="v1",
        )
        assert await commit_evaluation_and_complete_job(job_id, token, row) is True

        config = load_retrieval_config()
        pool = await get_pool(config)
        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = 'v1:judge-a' AND status = 'completed'
            """,
            event_id,
        )
        assert count == 1
        await close_pool()

    asyncio.run(_run())


def test_different_judge_models_same_prompt_create_distinct_evaluations() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        event_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        await insert_events(
            [
                {
                    "event_id": event_id,
                    "created_at": created_at,
                    "correlation_id": "dual-model",
                    "request_source": "pytest",
                    "status": "ok",
                    "request_payload": {"query": "q"},
                    "response_payload": {"results": []},
                    "retrieval_trace": {},
                    "metadata": {},
                }
            ]
        )

        for version, model in (("v1:model-a", "model-a"), ("v1:model-b", "model-b")):
            assert await enqueue_evaluation_job(event_id, created_at, version, max_attempts=3)
            jobs = await claim_evaluation_jobs(version, 1, 60, "worker")
            job = jobs[0]
            pool = await get_pool(config)
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

        pool = await get_pool(config)
        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND prompt_version = 'v1' AND status = 'completed'
            """,
            event_id,
        )
        assert count == 2
        await close_pool()

    asyncio.run(_run())


def test_reconciliation_respects_evaluation_version() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        event_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        await insert_events(
            [
                {
                    "event_id": event_id,
                    "created_at": created_at,
                    "correlation_id": "reconcile",
                    "request_source": "pytest",
                    "status": "ok",
                    "request_payload": {"query": "q"},
                    "response_payload": {"results": []},
                    "retrieval_trace": {},
                    "metadata": {},
                }
            ]
        )
        inserted = await reconcile_missing_evaluation_jobs("v2:new-model", limit=10, max_attempts=3)
        assert inserted == 1
        inserted_again = await reconcile_missing_evaluation_jobs("v2:new-model", limit=10, max_attempts=3)
        assert inserted_again == 0
        await close_pool()

    asyncio.run(_run())


def test_migrations_preserve_existing_evaluations() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        config = load_retrieval_config()
        await run_migrations(config)
        pool = await get_pool(config)
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
        await close_pool()

    asyncio.run(_run())


def test_concurrent_commits_only_one_succeeds() -> None:
    if _database_url() is None:
        pytest.skip("RETRIEVAL_DATABASE_URL is not set")

    async def _run() -> None:
        event_id, created_at, job_id, token = await _seed_event_and_job()
        row = _evaluation_row(
            event_id=event_id,
            created_at=created_at,
            evaluation_version="v1:judge-a",
            judge_model="judge-a",
            prompt_version="v1",
        )

        async def attempt() -> bool:
            return await commit_evaluation_and_complete_job(job_id, token, row)

        results = await asyncio.gather(attempt(), attempt())
        assert sum(1 for ok in results if ok) == 1

        config = load_retrieval_config()
        pool = await get_pool(config)
        count = await pool.fetchval(
            """
            SELECT COUNT(*)::int
            FROM retrieval_obs.retrieval_evaluations
            WHERE event_id = $1 AND evaluation_version = 'v1:judge-a'
            """,
            event_id,
        )
        assert count == 1
        await close_pool()

    asyncio.run(_run())
