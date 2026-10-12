"""Materialize full retrieval telemetry rows from pending queue payloads."""

from __future__ import annotations

import copy
from typing import Any

from radius_rag.observability.context import build_response_payload
from radius_rag.observability.sanitize import sanitize_payload
from radius_rag.observability.index_info import (
    chunk_code_hash,
    chunk_stable_id,
    get_active_index_info,
)
from radius_rag.observability.trace import build_retrieval_trace


def _enrich_candidates_for_trace(
    candidates: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    index_fingerprint: str | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        row = copy.deepcopy(candidate)
        item_index = row.get("metadata_index")
        if isinstance(item_index, int) and 0 <= item_index < len(metadata):
            item = metadata[item_index]
            if index_fingerprint:
                row["stable_id"] = chunk_stable_id(item, item_index, index_fingerprint)
            row["code_hash"] = chunk_code_hash(item)
            row["document_metadata"] = {
                key: item.get(key)
                for key in (
                    "repo",
                    "file",
                    "language",
                    "source_type",
                    "name",
                    "full_name",
                    "detail",
                    "start_line",
                    "end_line",
                    "git_url",
                    "updated_at",
                    "doc_hash",
                    "chunk_value",
                )
                if key in item
            }
        enriched.append(row)
    return enriched


def finalize_pending_event(pending: dict[str, Any]) -> dict[str, Any]:
    """Build full DB row from a queued pending snapshot (runs on writer thread)."""
    snapshot = pending.get("snapshot_input") or {}
    rag_service = snapshot.get("rag_service")
    query = snapshot.get("query", "")
    results = copy.deepcopy(snapshot.get("results") or [])
    trace_kwargs = copy.deepcopy(snapshot.get("trace_kwargs") or {})
    retrieval_config_snapshot = snapshot.get("retrieval_config_snapshot") or {}
    embedding_model = snapshot.get("embedding_model")

    index_info = get_active_index_info(rag_service)
    manifest = index_info.manifest or {}

    request_payload = sanitize_payload(snapshot.get("request_payload") or {})
    if not request_payload:
        request_payload = {"query": query}
    elif "query" not in request_payload:
        request_payload = {**request_payload, "query": query}

    response_base: dict[str, Any] = {
        "count": len(results),
        "results": results,
    }
    response_payload = build_response_payload(response_base)

    trace: dict[str, Any] = {}
    if trace_kwargs:
        candidates = trace_kwargs.get("candidates") or []
        metadata = getattr(rag_service, "metadata", []) or [] if rag_service else []
        enriched = _enrich_candidates_for_trace(
            candidates,
            metadata,
            index_info.index_fingerprint,
        )
        trace = build_retrieval_trace(
            query=query,
            faiss_ranks=trace_kwargs.get("faiss_ranks") or {},
            bm25_ranks=trace_kwargs.get("bm25_ranks") or {},
            faiss_scores=trace_kwargs.get("faiss_scores") or {},
            bm25_scores=trace_kwargs.get("bm25_scores") or {},
            candidates=enriched,
            results=results,
            retrieval_config=retrieval_config_snapshot,
            boosts=trace_kwargs.get("boosts"),
        )
        trace["chunk_texts"] = [
            {
                "metadata_index": row.get("metadata_index"),
                "stable_id": row.get("stable_id"),
                "code": row.get("code"),
                "code_hash": row.get("code_hash"),
                "document_metadata": row.get("document_metadata"),
            }
            for row in enriched
        ]

    base_metadata = dict(pending.get("metadata") or {})
    base_metadata.update(
        {
            "index_source": index_info.source,
            "manifest_built_at": manifest.get("built_at"),
            "manifest_checksums": manifest.get("checksums"),
            "document_count": index_info.document_count,
            "dimensions": index_info.dimensions,
        }
    )

    return {
        "event_id": pending.get("event_id"),
        "created_at": pending.get("created_at"),
        "correlation_id": pending.get("correlation_id"),
        "request_source": pending.get("request_source"),
        "index_staging_id": index_info.staging_id,
        "index_fingerprint": index_info.index_fingerprint,
        "embedding_model": index_info.embedding_model or embedding_model,
        "duration_ms": pending.get("duration_ms"),
        "status": pending.get("status", "ok"),
        "error_type": pending.get("error_type"),
        "request_payload": request_payload,
        "response_payload": response_payload,
        "retrieval_trace": trace,
        "metadata": base_metadata,
    }
