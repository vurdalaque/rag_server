"""Non-blocking retrieval event telemetry with a background DB writer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from radius_rag.observability.config import RetrievalConfig, load_retrieval_config
from radius_rag.observability.storage.db import (
    enqueue_evaluation_job,
    insert_events,
    reconcile_missing_evaluation_jobs,
)
from radius_rag.observability.snapshot import finalize_pending_event
from radius_rag.observability.telemetry.metrics import (
    record_batch_written,
    record_drop,
    record_enqueue_error,
    record_event_bytes,
    record_events_written,
    record_queue_depth,
    record_write_error,
)

logger = logging.getLogger(__name__)

_queue: asyncio.Queue[dict[str, Any] | None] | None = None
_writer_task: asyncio.Task[None] | None = None
_config: RetrievalConfig | None = None
_stop_event: asyncio.Event | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _approx_event_bytes(event: dict[str, Any]) -> int:
    try:
        return len(json.dumps(event, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _utcnow()


async def _enqueue_jobs_for_events(
    events: list[dict[str, Any]],
    cfg: RetrievalConfig,
) -> None:
    for event in events:
        event_id = event.get("event_id")
        created_at = event.get("created_at")
        if event_id is None or created_at is None:
            continue
        try:
            await enqueue_evaluation_job(
                UUID(str(event_id)) if not isinstance(event_id, UUID) else event_id,
                _parse_created_at(created_at),
                cfg.evaluation_version,
                max_attempts=cfg.max_retries + 1,
            )
        except Exception as error:
            record_enqueue_error(type(error).__name__)
            logger.exception(
                "Failed to enqueue evaluation job for event %s (retrieval event already saved)",
                event_id,
            )


async def _materialize_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for item in batch:
        if item.get("_pending_snapshot"):
            try:
                materialized.append(finalize_pending_event(item))
            except Exception:
                logger.exception("Failed to materialize retrieval snapshot")
                continue
        else:
            materialized.append(item)
    return materialized


async def _flush_batch(batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    cfg = _config or load_retrieval_config()

    materialized = await _materialize_batch(batch)
    if not materialized:
        return

    try:
        written = await insert_events(materialized)
        record_events_written(written)
        record_batch_written()
    except Exception as error:
        record_write_error(type(error).__name__)
        logger.exception("Failed to persist retrieval telemetry batch (%s events)", len(materialized))
        return

    saved = [event for event in materialized if event.get("event_id")]
    if cfg.evaluation_auto_enqueue:
        try:
            await _enqueue_jobs_for_events(saved, cfg)
        except Exception as error:
            record_enqueue_error(type(error).__name__)
            logger.exception(
                "Evaluation job enqueue failed after persisting %s events",
                len(saved),
            )

    try:
        await reconcile_missing_evaluation_jobs(
            cfg.evaluation_version,
            limit=max(cfg.batch_size * 5, 50),
            max_attempts=cfg.max_retries + 1,
        )
    except Exception:
        logger.exception("Evaluation job reconciliation failed")


async def start_retrieval_telemetry(config: RetrievalConfig | None = None) -> None:
    global _queue, _writer_task, _config, _stop_event
    cfg = config or load_retrieval_config()
    if not cfg.telemetry_ready():
        logger.info("Retrieval telemetry disabled or database URL missing")
        return
    if _writer_task is not None and not _writer_task.done():
        return

    _config = cfg
    _stop_event = asyncio.Event()
    _queue = asyncio.Queue(maxsize=cfg.queue_size)
    _writer_task = asyncio.create_task(_writer_loop(), name="retrieval-telemetry-writer")
    logger.info("Retrieval telemetry writer started")


async def stop_retrieval_telemetry(drain: bool = True) -> None:
    global _queue, _writer_task, _config, _stop_event
    if _queue is None or _writer_task is None:
        return

    if drain:
        try:
            _queue.put_nowait(None)
        except asyncio.QueueFull:
            await _queue.put(None)
    elif _stop_event is not None:
        _stop_event.set()

    try:
        await asyncio.wait_for(_writer_task, timeout=30.0)
    except asyncio.TimeoutError:
        _worker_task_cancel = _writer_task
        _worker_task_cancel.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _worker_task_cancel

    _queue = None
    _writer_task = None
    _config = None
    _stop_event = None
    record_queue_depth(0)


def emit(event: dict[str, Any]) -> bool:
    """Enqueue a telemetry event without blocking the caller."""
    if _queue is None:
        return False

    payload = dict(event)
    if payload.get("event_id") is None:
        payload["event_id"] = str(uuid4())
    if payload.get("created_at") is None:
        payload["created_at"] = _utcnow().isoformat()

    try:
        _queue.put_nowait(payload)
    except asyncio.QueueFull:
        record_drop()
        logger.warning("Retrieval telemetry queue full; dropping event %s", payload.get("event_id"))
        return False

    record_event_bytes(_approx_event_bytes(payload))
    record_queue_depth(_queue.qsize())
    return True


async def _writer_loop() -> None:
    cfg = _config or load_retrieval_config()
    assert _queue is not None

    batch: list[dict[str, Any]] = []
    flush_interval = cfg.flush_interval_seconds

    while True:
        if _stop_event is not None and _stop_event.is_set() and _queue.empty():
            if batch:
                await _flush_batch(batch)
            break

        try:
            item = await asyncio.wait_for(_queue.get(), timeout=flush_interval)
        except asyncio.TimeoutError:
            if batch:
                await _flush_batch(batch)
                batch = []
            record_queue_depth(_queue.qsize())
            continue

        if item is None:
            if batch:
                await _flush_batch(batch)
            while not _queue.empty():
                leftover = _queue.get_nowait()
                if leftover is not None:
                    batch.append(leftover)
            if batch:
                await _flush_batch(batch)
            break

        batch.append(item)
        record_queue_depth(_queue.qsize())
        if len(batch) >= cfg.batch_size:
            await _flush_batch(batch)
            batch = []
