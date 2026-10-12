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
from radius_rag.observability.storage._asyncpg import load_asyncpg
from radius_rag.observability.storage.db import (
    EvaluationCommitLeaseError,
    claim_evaluation_jobs,
    close_pool,
    commit_evaluation_and_complete_job,
    extend_evaluation_job_lease,
    fetch_retrieval_event,
    finish_evaluation_job,
    mark_evaluation_job_running,
    reconcile_missing_evaluation_jobs,
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


def _asyncpg_transient_types() -> tuple[type[BaseException], ...]:
    try:
        asyncpg = load_asyncpg()
    except RuntimeError:
        return ()
    types: list[type[BaseException]] = []
    for name in (
        "PostgresConnectionError",
        "InterfaceError",
        "CannotConnectNowError",
        "ConnectionDoesNotExistError",
    ):
        cls = getattr(asyncpg, name, None)
        if isinstance(cls, type):
            types.append(cls)
    return tuple(types)


def _is_retryable_infra_error(error: BaseException) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError)):
        return True
    if isinstance(error, OSError):
        return True
    pg_types = _asyncpg_transient_types()
    return bool(pg_types) and isinstance(error, pg_types)


def _is_programming_error(error: BaseException) -> bool:
    if isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
        return True
    return False


async def _release_job_for_retry(
    job: dict[str, Any],
    job_id: int,
    lease_token: UUID,
    config: RetrievalConfig,
    error: str,
) -> None:
    attempts = int(job.get("attempts", 1))
    max_attempts = int(job.get("max_attempts", config.max_retries + 1))
    if attempts < max_attempts:
        delay = _backoff_seconds(config, attempts)
        next_at = _utcnow() + timedelta(seconds=delay)
        updated = await finish_evaluation_job(
            job_id,
            lease_token,
            status="pending",
            error=error,
            next_attempt_at=next_at,
        )
        if not updated:
            logger.warning("Job %s lease lost while scheduling retry", job_id)
        return
    updated = await finish_evaluation_job(
        job_id,
        lease_token,
        status="dead",
        error=error,
    )
    if updated:
        logger.error("Evaluation job %s exhausted attempts: %s", job_id, error)


async def _process_job(
    job: dict[str, Any],
    config: RetrievalConfig,
    http_client: httpx.AsyncClient,
) -> None:
    job_id = int(job["job_id"])
    lease_token = job["lease_token"]
    if not isinstance(lease_token, UUID):
        lease_token = UUID(str(lease_token))

    try:
        if not await mark_evaluation_job_running(job_id, lease_token, config.lease_ttl_seconds):
            logger.warning("Job %s lease token rejected when marking running", job_id)
            return

        event = await fetch_retrieval_event(job["event_id"], job["event_created_at"])
    except Exception as error:
        if _is_retryable_infra_error(error):
            logger.warning("Infrastructure error before job %s processing: %s", job_id, error)
            await close_pool()
            return
        raise

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
        if _is_retryable_infra_error(error):
            await _release_job_for_retry(job, job_id, lease_token, config, str(error))
            return
        if _is_programming_error(error):
            await finish_evaluation_job(
                job_id,
                lease_token,
                status="failed",
                error=str(error),
            )
            logger.exception("Evaluation job %s failed (non-retryable): %s", job_id, error)
            return
        await _release_job_for_retry(job, job_id, lease_token, config, str(error))
        logger.exception("Evaluation job %s failed (retryable)", job_id)
        return

    lease_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await lease_task

    if result.get("status") == "incomplete":
        await finish_evaluation_job(
            job_id,
            lease_token,
            status="failed",
            error=str(result.get("diagnostics")),
        )
        return

    if result.get("status") != "completed":
        await finish_evaluation_job(
            job_id,
            lease_token,
            status="failed",
            error=str(result.get("diagnostics")),
        )
        return

    try:
        committed = await commit_evaluation_and_complete_job(
            job_id,
            lease_token,
            {
                "event_id": job["event_id"],
                "event_created_at": job["event_created_at"],
                "evaluation_version": job.get("evaluation_version"),
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
            },
        )
    except EvaluationCommitLeaseError as error:
        logger.warning("Job %s commit rejected (lease): %s", job_id, error)
        return
    except Exception as error:
        if _is_retryable_infra_error(error):
            logger.warning(
                "Unknown commit outcome for job %s after infra error: %s",
                job_id,
                error,
            )
            await close_pool()
            return
        raise

    if not committed:
        logger.warning(
            "Job %s result discarded (stale lease or job reclaimed)",
            job_id,
        )


async def _lease_heartbeat(job_id: int, lease_token: UUID, config: RetrievalConfig) -> None:
    interval = max(5.0, config.lease_ttl_seconds / 3.0)
    while True:
        await asyncio.sleep(interval)
        try:
            extended = await extend_evaluation_job_lease(
                job_id,
                lease_token,
                config.lease_ttl_seconds,
            )
        except Exception as error:
            if _is_retryable_infra_error(error):
                logger.warning("Lease heartbeat failed for job %s: %s", job_id, error)
                break
            raise
        if not extended:
            break


async def _worker_loop(config: RetrievalConfig, stop_event: asyncio.Event) -> None:
    global _semaphore
    _semaphore = asyncio.Semaphore(config.evaluator_concurrency)
    worker_id = f"worker-{id(asyncio.current_task())}"
    infra_backoff = config.poll_interval_seconds

    timeout = httpx.Timeout(config.judge_llm_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        while not stop_event.is_set():
            try:
                await reconcile_missing_evaluation_jobs(
                    config.evaluation_version,
                    limit=max(config.batch_size * 5, 50),
                    max_attempts=config.max_retries + 1,
                )
                jobs = await claim_evaluation_jobs(
                    config.evaluation_version,
                    limit=1,
                    lease_ttl_seconds=config.lease_ttl_seconds,
                    worker_id=worker_id,
                )
                infra_backoff = config.poll_interval_seconds
            except Exception as error:
                if _is_retryable_infra_error(error):
                    logger.warning("Evaluator DB unavailable: %s", error)
                    await close_pool()
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=infra_backoff)
                    except asyncio.TimeoutError:
                        infra_backoff = min(infra_backoff * 2, 60.0)
                    continue
                logger.exception("Unexpected error claiming evaluation jobs")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=config.poll_interval_seconds)
                except asyncio.TimeoutError:
                    continue
                continue

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
