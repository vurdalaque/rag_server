"""Background worker that runs retrieval judge evaluations."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import httpx

from radius_rag.observability.config import RetrievalConfig, load_retrieval_config
from radius_rag.observability.storage.db import (
    claim_evaluation_jobs,
    extend_evaluation_job_lease,
    fetch_retrieval_event,
    finish_evaluation_job,
    insert_evaluation,
    mark_evaluation_job_running,
)
from radius_rag.observability.evaluation.judge import run_judge_for_event

logger = logging.getLogger(__name__)

_worker_task: asyncio.Task[None] | None = None
_stop_event: asyncio.Event | None = None
_semaphore: asyncio.Semaphore | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _backoff_seconds(config: RetrievalConfig, attempt: int) -> float:
    base = config.retry_base_seconds * (2 ** max(0, attempt - 1))
    return min(base, 3600.0)


async def _process_job(
    job: dict[str, Any],
    config: RetrievalConfig,
    http_client: httpx.AsyncClient,
) -> None:
    job_id = int(job["job_id"])
    lease_token = job["lease_token"]
    if not isinstance(lease_token, UUID):
        lease_token = UUID(str(lease_token))

    if not await mark_evaluation_job_running(job_id, lease_token, config.lease_ttl_seconds):
        logger.warning("Job %s lease token rejected when marking running", job_id)
        return

    event = await fetch_retrieval_event(job["event_id"], job["event_created_at"])
    if event is None:
        await finish_evaluation_job(
            job_id,
            lease_token,
            status="failed",
            error="retrieval event not found",
        )
        return

    lease_task = asyncio.create_task(_lease_heartbeat(job_id, lease_token, config))

    try:
        result = await run_judge_for_event(dict(event), config, client=http_client)
    except Exception as error:
        lease_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await lease_task
        attempts = int(job.get("attempts", 1))
        max_attempts = int(job.get("max_attempts", config.max_retries + 1))
        if attempts < max_attempts:
            delay = _backoff_seconds(config, attempts)
            next_at = _utcnow() + timedelta(seconds=delay)
            await finish_evaluation_job(
                job_id,
                lease_token,
                status="pending",
                error=str(error),
                next_attempt_at=next_at,
            )
            return
        await finish_evaluation_job(
            job_id,
            lease_token,
            status="dead",
            error=str(error),
        )
        logger.exception("Evaluation job %s failed permanently", job_id)
        return

    lease_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await lease_task

    if result.get("status") != "completed":
        await finish_evaluation_job(
            job_id,
            lease_token,
            status="failed",
            error=str(result.get("diagnostics")),
        )
        return

    await insert_evaluation(
        {
            "event_id": job["event_id"],
            "event_created_at": job["event_created_at"],
            "judge_model": result["judge_model"],
            "prompt_version": result["prompt_version"],
            "per_chunk_scores": result["per_chunk_scores"],
            "context_sufficiency": result["context_sufficiency"],
            "query_classification": result["query_classification"],
            "aggregated_metrics": result["aggregated_metrics"],
            "explanation": result["explanation"],
            "status": result["status"],
            "diagnostics": result["diagnostics"],
            "truncation_info": result["truncation_info"],
        }
    )
    await finish_evaluation_job(job_id, lease_token, status="completed", error=None)


async def _lease_heartbeat(job_id: int, lease_token: UUID, config: RetrievalConfig) -> None:
    interval = max(5.0, config.lease_ttl_seconds / 3.0)
    while True:
        await asyncio.sleep(interval)
        extended = await extend_evaluation_job_lease(
            job_id,
            lease_token,
            config.lease_ttl_seconds,
        )
        if not extended:
            break


async def _worker_loop(config: RetrievalConfig, stop_event: asyncio.Event) -> None:
    global _semaphore
    _semaphore = asyncio.Semaphore(config.evaluator_concurrency)
    worker_id = f"worker-{id(asyncio.current_task())}"

    timeout = httpx.Timeout(config.judge_llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        while not stop_event.is_set():
            jobs = await claim_evaluation_jobs(
                config.evaluation_version,
                limit=1,
                lease_ttl_seconds=config.lease_ttl_seconds,
                worker_id=worker_id,
            )
            if not jobs:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=config.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    continue
                continue

            for job in jobs:
                async with _semaphore:
                    if stop_event.is_set():
                        break
                    await _process_job(job, config, http_client)


async def start_evaluation_worker(config: RetrievalConfig | None = None) -> asyncio.Event:
    global _worker_task, _stop_event
    cfg = config or load_retrieval_config()
    if not cfg.database_url:
        raise RuntimeError("RETRIEVAL_DATABASE_URL is not configured")

    if _worker_task is not None and not _worker_task.done():
        assert _stop_event is not None
        return _stop_event

    _stop_event = asyncio.Event()
    _worker_task = asyncio.create_task(
        _worker_loop(cfg, _stop_event),
        name="retrieval-eval-worker",
    )
    return _stop_event


async def stop_evaluation_worker(stop_event: asyncio.Event | None = None) -> None:
    global _worker_task, _stop_event
    event = stop_event or _stop_event
    if event is None:
        return
    event.set()
    if _worker_task is not None:
        try:
            await asyncio.wait_for(_worker_task, timeout=60.0)
        except asyncio.TimeoutError:
            _worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _worker_task
        _worker_task = None
    _stop_event = None
