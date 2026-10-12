from __future__ import annotations

from retrieval_candidate import (
    candidates_from_response_payload,
    candidates_from_trace,
    extract_chunk_text,
    normalize_candidate,
)


def test_extract_chunk_text_prefers_code() -> None:
    text, present, field = extract_chunk_text({"code": "hello", "text": "other"})
    assert present is True
    assert field == "code"
    assert text == "hello"


def test_extract_chunk_text_falls_back_to_text() -> None:
    text, present, field = extract_chunk_text({"text": "body"})
    assert present and field == "text" and text == "body"


def test_candidates_from_response_results() -> None:
    items = candidates_from_response_payload(
        {"results": [{"metadata_index": 1, "code": "x", "score": 0.5}]}
    )
    assert len(items) == 1
    assert items[0].text == "x"
    assert items[0].rank == 1


def test_candidates_from_trace_includes_pool_outside_top_k() -> None:
    trace = {
        "candidates": [
            {"metadata_index": 0, "in_top_k": True, "top_k_rank": 1, "code": "a"},
            {"metadata_index": 1, "in_top_k": False, "code": "b"},
        ],
        "chunk_texts": [
            {"metadata_index": 1, "code": "b", "code_hash": "abc"},
        ],
    }
    items = candidates_from_trace(trace)
    assert len(items) == 2
    pool = next(c for c in items if c.metadata_index == 1)
    assert pool.text == "b"
    assert pool.rank is None
