"""Prometheus metrics for retrieval telemetry persistence."""

from __future__ import annotations

from prometheus_client import Counter, Gauge

TELEMETRY_QUEUE_DEPTH = Gauge(
    "retrieval_telemetry_queue_depth",
    "Current depth of the in-memory retrieval telemetry queue.",
)
TELEMETRY_QUEUE_DROPS = Counter(
    "retrieval_telemetry_queue_drops_total",
    "Retrieval telemetry events dropped because the queue was full.",
)
TELEMETRY_EVENT_BYTES = Counter(
    "retrieval_telemetry_event_bytes_total",
    "Approximate serialized bytes of retrieval telemetry events accepted.",
)
TELEMETRY_WRITE_ERRORS = Counter(
    "retrieval_telemetry_write_errors_total",
    "Errors while flushing retrieval telemetry batches to the database.",
    ["error_type"],
)
TELEMETRY_EVENTS_WRITTEN = Counter(
    "retrieval_telemetry_events_written_total",
    "Retrieval telemetry events successfully persisted.",
)
TELEMETRY_BATCHES_WRITTEN = Counter(
    "retrieval_telemetry_batches_written_total",
    "Retrieval telemetry batch flush operations completed.",
)
TELEMETRY_ENQUEUE_ERRORS = Counter(
    "retrieval_telemetry_enqueue_errors_total",
    "Evaluation job enqueue failures after successful event persistence.",
    ["error_type"],
)


def record_queue_depth(depth: int) -> None:
    TELEMETRY_QUEUE_DEPTH.set(max(0, depth))


def record_drop() -> None:
    TELEMETRY_QUEUE_DROPS.inc()


def record_event_bytes(byte_count: int) -> None:
    if byte_count > 0:
        TELEMETRY_EVENT_BYTES.inc(byte_count)


def record_write_error(error_type: str) -> None:
    TELEMETRY_WRITE_ERRORS.labels(error_type=error_type).inc()


def record_events_written(count: int) -> None:
    if count > 0:
        TELEMETRY_EVENTS_WRITTEN.inc(count)


def record_batch_written() -> None:
    TELEMETRY_BATCHES_WRITTEN.inc()


def record_enqueue_error(error_type: str) -> None:
    TELEMETRY_ENQUEUE_ERRORS.labels(error_type=error_type).inc()
