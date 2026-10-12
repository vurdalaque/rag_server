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


def _metadata_indices(*groups: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for group in groups:
        for row in group:
            if not isinstance(row, dict):
                continue
            idx = row.get("metadata_index")
            if isinstance(idx, int):
                indices.add(idx)
    return indices


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
            if row.get("code") in (None, "") and isinstance(item.get("code"), str):
                row["code"] = item["code"]
        enriched.append(row)
    return enriched


def prepare_frozen_snapshot_input(
    *,
    rag_service: Any,
    query: str,
    results: list[dict[str, Any]],
    trace_kwargs: dict[str, Any],
    request_payload: dict[str, Any],
    retrieval_config_snapshot: dict[str, Any],
    embedding_model: str | None,
) -> dict[str, Any]:
    """Capture index version and chunk text at retrieval time (before async DB write)."""
    index_info = get_active_index_info(rag_service)
    manifest = index_info.manifest or {}
    metadata = getattr(rag_service, "metadata", []) or []

    results_copy = copy.deepcopy(results)
    trace_copy = copy.deepcopy(trace_kwargs)
    candidates = trace_copy.get("candidates") or []
    enriched = _enrich_candidates_for_trace(
        candidates,
        metadata,
        index_info.index_fingerprint,
    )
    trace_copy["candidates"] = enriched

    used_indices = _metadata_indices(results_copy, enriched)
    metadata_snapshot: dict[int, dict[str, Any]] = {}
    for idx in used_indices:
        if 0 <= idx < len(metadata):
            metadata_snapshot[idx] = {
                key: metadata[idx].get(key)
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
                    "code",
                )
                if key in metadata[idx]
            }

    return {
        "query": query,
        "results": results_copy,
        "trace_kwargs": trace_copy,
        "request_payload": request_payload,
        "retrieval_config_snapshot": retrieval_config_snapshot,
        "embedding_model": embedding_model,
        "index_info": {
            "staging_id": index_info.staging_id,
            "index_fingerprint": index_info.index_fingerprint,
            "embedding_model": index_info.embedding_model,
            "source": index_info.source,
            "document_count": index_info.document_count,
            "dimensions": index_info.dimensions,
            "manifest": copy.deepcopy(manifest),
        },
        "metadata_snapshot": metadata_snapshot,
    }


def finalize_pending_event(pending: dict[str, Any]) -> dict[str, Any]:
    """Build full DB row from a queued pending snapshot (runs on writer thread)."""
    snapshot = pending.get("snapshot_input") or {}
    query = snapshot.get("query", "")
    results = copy.deepcopy(snapshot.get("results") or [])
    trace_kwargs = copy.deepcopy(snapshot.get("trace_kwargs") or {})
    retrieval_config_snapshot = snapshot.get("retrieval_config_snapshot") or {}
    embedding_model = snapshot.get("embedding_model")
    index_info_data = snapshot.get("index_info") or {}
    manifest = index_info_data.get("manifest") or {}

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
        enriched = trace_kwargs.get("candidates") or []
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
            "index_source": index_info_data.get("source"),
            "manifest_built_at": manifest.get("built_at"),
            "manifest_checksums": manifest.get("checksums"),
            "document_count": index_info_data.get("document_count"),
            "dimensions": index_info_data.get("dimensions"),
        }
    )

    return {
        "event_id": pending.get("event_id"),
        "created_at": pending.get("created_at"),
        "correlation_id": pending.get("correlation_id"),
        "request_source": pending.get("request_source"),
        "index_staging_id": index_info_data.get("staging_id"),
        "index_fingerprint": index_info_data.get("index_fingerprint"),
        "embedding_model": index_info_data.get("embedding_model") or embedding_model,
        "duration_ms": pending.get("duration_ms"),
        "status": pending.get("status", "ok"),
        "error_type": pending.get("error_type"),
        "request_payload": request_payload,
        "response_payload": response_payload,
        "retrieval_trace": trace,
        "metadata": base_metadata,
    }
