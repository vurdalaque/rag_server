"""Measure snapshot freeze overhead on realistic candidate pool sizes."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from radius_rag.observability.snapshot import prepare_frozen_snapshot_input

def _large_metadata(count: int, chunk_chars: int) -> list[dict]:
    body = "x" * chunk_chars
    return [{"code": body, "file": f"f{i}.py", "repo": "r"} for i in range(count)]


def test_snapshot_freeze_overhead_realistic_pool() -> None:
    """Document overhead; threshold is generous to avoid flaky CI."""
    candidate_count = 80
    chunk_chars = 4096
    metadata = _large_metadata(candidate_count, chunk_chars)
    candidates = [
        {"metadata_index": i, "score": 1.0 - i * 0.001, "in_top_k": i < 10, "top_k_rank": i + 1}
        for i in range(candidate_count)
    ]
    results = [{"metadata_index": i, "score": 1.0} for i in range(10)]

    rag = MagicMock()
    rag.metadata = metadata

    with patch(
        "radius_rag.observability.snapshot.get_active_index_info",
        return_value=MagicMock(
            staging_id="s",
            index_fingerprint="fp",
            embedding_model="e",
            source="local",
            document_count=candidate_count,
            dimensions=384,
            manifest={},
        ),
    ):
        start = time.perf_counter()
        for _ in range(5):
            prepare_frozen_snapshot_input(
                rag_service=rag,
                query="q" * 200,
                results=results,
                trace_kwargs={"candidates": candidates},
                request_payload={"query": "q"},
                retrieval_config_snapshot={},
                embedding_model="e",
            )
        elapsed_ms = (time.perf_counter() - start) / 5 * 1000

    # deepcopy of ~80 x 4KB candidates dominates; expect sub-second on dev hardware.
    assert elapsed_ms < 2000.0, f"snapshot freeze took {elapsed_ms:.1f}ms (consider shallow freeze if production p99 suffers)"
