"""Shared helpers for integration tests against a shared retrieval_obs schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from retrieval_config import load_retrieval_config
from retrieval_db import claim_evaluation_jobs, enqueue_evaluation_job, get_pool, insert_events


def new_evaluation_version(prefix: str) -> str:
    return f"pytest-{prefix}-{uuid.uuid4()}"


async def insert_test_event(correlation_id: str) -> tuple[uuid.UUID, datetime]:
    event_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    await insert_events(
        [
            {
                "event_id": event_id,
                "created_at": created_at,
                "correlation_id": correlation_id,
                "request_source": "pytest",
                "status": "ok",
                "request_payload": {"query": "q"},
                "response_payload": {"results": []},
                "retrieval_trace": {},
                "metadata": {},
            }
        ]
    )
    return event_id, created_at


async def seed_event_and_running_job(
    *,
    prefix: str,
    max_attempts: int = 3,
    worker_id: str = "pytest-worker",
    attempts: int | None = None,
) -> tuple[uuid.UUID, datetime, int, uuid.UUID, str]:
    """Unique evaluation_version per call — safe on shared app_db."""
    evaluation_version = new_evaluation_version(prefix)
    event_id, created_at = await insert_test_event(prefix)
    assert await enqueue_evaluation_job(
        event_id,
        created_at,
        evaluation_version,
        max_attempts=max_attempts,
    )
    jobs = await claim_evaluation_jobs(
        evaluation_version,
        1,
        300,
        worker_id,
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert job["event_id"] == event_id
    job_id = int(job["job_id"])
    token = job["lease_token"]
    attempt_count = attempts if attempts is not None else int(job["attempts"])
    config = load_retrieval_config()
    pool = await get_pool(config)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET status = 'running',
                attempts = $2,
                lease_expires_at = now() + interval '10 minutes'
            WHERE job_id = $1
            """,
            job_id,
            attempt_count,
        )
    return event_id, created_at, job_id, token, evaluation_version
