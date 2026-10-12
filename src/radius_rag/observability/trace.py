"""Structured retrieval trace payloads for observability."""

from __future__ import annotations

from typing import Any

from radius_rag.observability.index_info import RRF_CONSTANT


def _rrf_contribution(rank: int | None, constant: int = RRF_CONSTANT) -> float | None:
    if rank is None:
        return None
    return 1.0 / (constant + rank)


def _candidate_trace_row(
    candidate: dict[str, Any],
    *,
    faiss_rank: int | None,
    bm25_rank: int | None,
    in_top_k: bool,
    top_k_rank: int | None,
) -> dict[str, Any]:
    return {
        "metadata_index": candidate.get("metadata_index"),
        "stable_id": candidate.get("stable_id"),
        "file": candidate.get("file"),
        "repo": candidate.get("repo"),
        "source_type": candidate.get("source_type"),
        "faiss_score": candidate.get("faiss_score"),
        "bm25_score": candidate.get("bm25_score"),
        "hybrid_score": candidate.get("hybrid_score"),
        "rerank_score": candidate.get("rerank_score"),
        "score": candidate.get("score"),
        "faiss_rank": faiss_rank,
        "bm25_rank": bm25_rank,
        "rrf_faiss": _rrf_contribution(faiss_rank),
        "rrf_bm25": _rrf_contribution(bm25_rank),
        "recency_factor": candidate.get("recency_factor"),
        "chunk_value_factor": candidate.get("chunk_value_factor"),
        "in_top_k": in_top_k,
        "top_k_rank": top_k_rank,
        "code": candidate.get("code"),
        "code_hash": candidate.get("code_hash"),
        "document_metadata": candidate.get("document_metadata") or {},
        "final_rank": top_k_rank,
    }


def build_retrieval_trace(
    *,
    query: str,
    faiss_ranks: dict[int, int],
    bm25_ranks: dict[int, int],
    faiss_scores: dict[int, float],
    bm25_scores: dict[int, float],
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    retrieval_config: dict[str, Any],
    boosts: dict[str, Any] | None = None,
    rrf_constant: int = RRF_CONSTANT,
) -> dict[str, Any]:
    top_k_indices: list[int] = []
    for result in results:
        idx = result.get("metadata_index")
        if isinstance(idx, int):
            top_k_indices.append(idx)
    top_k_set = set(top_k_indices)
    top_k_rank_map = {idx: rank for rank, idx in enumerate(top_k_indices, start=1)}

    candidate_rows: list[dict[str, Any]] = []
    pool_not_in_top_k: list[dict[str, Any]] = []

    for candidate in candidates:
        item_index = candidate.get("metadata_index")
        if not isinstance(item_index, int):
            continue
        faiss_rank = faiss_ranks.get(item_index)
        bm25_rank = bm25_ranks.get(item_index)
        in_top_k = item_index in top_k_set
        row = _candidate_trace_row(
            candidate,
            faiss_rank=faiss_rank,
            bm25_rank=bm25_rank,
            in_top_k=in_top_k,
            top_k_rank=top_k_rank_map.get(item_index),
        )
        candidate_rows.append(row)
        if not in_top_k:
            pool_not_in_top_k.append(row)

    candidate_rows.sort(
        key=lambda row: (
            row.get("top_k_rank") if row.get("in_top_k") else 10_000,
            -(row.get("score") or 0.0),
        ),
    )

    return {
        "query": query,
        "rrf_constant": rrf_constant,
        "boosts": boosts or {},
        "retrieval_config": retrieval_config,
        "faiss_scores": {str(k): v for k, v in faiss_scores.items()},
        "bm25_scores": {str(k): v for k, v in bm25_scores.items()},
        "faiss_ranks": {str(k): v for k, v in faiss_ranks.items()},
        "bm25_ranks": {str(k): v for k, v in bm25_ranks.items()},
        "candidate_count": len(candidate_rows),
        "top_k_count": len(top_k_indices),
        "candidates": candidate_rows,
        "pool_not_in_top_k": pool_not_in_top_k,
        "top_k_metadata_indices": top_k_indices,
    }
