"""Build and enqueue retrieval telemetry events from pipeline outcomes."""

from __future__ import annotations

import copy
from typing import Any

from radius_rag.observability.config import TELEMETRY_ENABLED, load_retrieval_config
from radius_rag.observability.context import (
    get_correlation_id,
    get_request_id,
    get_request_payload,
    get_request_source,
    get_session_id,
    new_request_id,
)
from radius_rag.observability.sanitize import sanitize_payload
from radius_rag.observability.telemetry.writer import emit


def maybe_emit_retrieval_event(
    *,
    rag_service: Any,
    query: str,
    duration_ms: int,
    status: str = "ok",
    error_type: str | None = None,
    results: list[dict[str, Any]] | None = None,
    search_params: dict[str, Any] | None = None,
    trace_kwargs: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    retrieval_config_snapshot: dict[str, Any] | None = None,
    embedding_model: str | None = None,
) -> None:
    if not TELEMETRY_ENABLED:
        return

    cfg = load_retrieval_config()
    if not cfg.telemetry_ready():
        return

    request_payload = sanitize_payload(get_request_payload())
    if search_params:
        request_payload = {**request_payload, "search_params": search_params}

    correlation_id = get_correlation_id() or get_request_id() or new_request_id()
    metadata: dict[str, Any] = {
        "request_id": get_request_id() or new_request_id(),
        "session_id": get_session_id(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    pending = {
        "_pending_snapshot": True,
        "correlation_id": correlation_id,
        "request_source": get_request_source(),
        "duration_ms": duration_ms,
        "status": status,
        "error_type": error_type,
        "metadata": metadata,
        "snapshot_input": {
            "rag_service": rag_service,
            "query": query,
            "results": copy.deepcopy(results or []),
            "trace_kwargs": copy.deepcopy(trace_kwargs or {}),
            "request_payload": request_payload,
            "retrieval_config_snapshot": retrieval_config_snapshot or {},
            "embedding_model": embedding_model,
        },
    }
    emit(pending)
