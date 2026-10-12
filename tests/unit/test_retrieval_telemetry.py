from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = MagicMock()

from retrieval_event import maybe_emit_retrieval_event, sanitize_payload
from retrieval_telemetry import emit, start_retrieval_telemetry, stop_retrieval_telemetry
from retrieval_trace import build_retrieval_trace


def test_build_retrieval_trace_rrf_fields() -> None:
    candidates = [
        {"metadata_index": 0, "score": 0.9, "file": "a.py"},
        {"metadata_index": 1, "score": 0.4, "file": "b.py"},
    ]
    results = [{"metadata_index": 0}]
    trace = build_retrieval_trace(
        query="needle",
        faiss_ranks={0: 1, 1: 2},
        bm25_ranks={1: 1, 0: 2},
        faiss_scores={0: 0.8, 1: 0.2},
        bm25_scores={0: 1.0, 1: 3.0},
        candidates=candidates,
        results=results,
        retrieval_config={"max_top_k": 6},
    )
    assert trace["query"] == "needle"
    assert trace["top_k_count"] == 1
    assert trace["candidates"][0]["in_top_k"] is True
    assert trace["candidates"][0]["faiss_rank"] == 1
    assert trace["candidates"][1]["in_top_k"] is False
    assert trace["candidates"][0]["rrf_faiss"] == pytest.approx(1 / 61)


def test_sanitize_payload_strips_auth_headers() -> None:
    payload = {
        "query": "find auth flow",
        "headers": {"Authorization": "Bearer secret", "X-Request-ID": "abc"},
        "authorization": "should drop",
    }
    cleaned = sanitize_payload(payload)
    assert cleaned["query"] == "find auth flow"
    assert "authorization" not in cleaned
    assert "Authorization" not in cleaned["headers"]


def test_emit_queue_full_increments_drop_counter() -> None:
    import asyncio
    from retrieval_config import MIGRATIONS_DIR, RetrievalConfig
    from retrieval_telemetry_metrics import TELEMETRY_QUEUE_DROPS

    before = TELEMETRY_QUEUE_DROPS._value.get()  # type: ignore[attr-defined]

    cfg = RetrievalConfig(
        telemetry_enabled=True,
        evaluation_auto_enqueue=False,
        database_url="postgresql://example",
        queue_size=1,
        batch_size=10,
        flush_interval_seconds=0.05,
        evaluator_concurrency=1,
        lease_ttl_seconds=30,
        max_retries=1,
        retry_base_seconds=1.0,
        poll_interval_seconds=1.0,
        judge_prompt_version="v1",
        judge_llm_url="http://127.0.0.1:8001/v1/chat/completions",
        judge_llm_model="judge",
        judge_llm_temperature=0.0,
        judge_llm_top_p=1.0,
        judge_llm_top_k=20,
        judge_llm_presence_penalty=0.0,
        judge_llm_repetition_penalty=1.0,
        judge_llm_enable_thinking=False,
        judge_llm_timeout_seconds=30.0,
        compression_config_key="key",
        judge_relevance_threshold=2,
        judge_max_query_chars=8000,
        judge_max_chunk_chars=6000,
        judge_max_chunks=30,
        judge_json_max_retries=2,
        migrations_dir=MIGRATIONS_DIR,
    )

    async def _run() -> None:
        with patch(
            "radius_rag.observability.telemetry.writer.insert_events",
            new_callable=AsyncMock,
        ):
            await start_retrieval_telemetry(cfg)
            assert emit({"event_id": "1"}) is True
            assert emit({"event_id": "2"}) is False
            await stop_retrieval_telemetry(drain=False)

    asyncio.run(_run())

    after = TELEMETRY_QUEUE_DROPS._value.get()  # type: ignore[attr-defined]
    assert after >= before + 1


def test_maybe_emit_does_not_call_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve_mock = AsyncMock(return_value=[{"metadata_index": 0}])
    monkeypatch.setattr("radius_rag.observability.event.TELEMETRY_ENABLED", True)

    cfg = MagicMock()
    cfg.telemetry_ready.return_value = True
    cfg.evaluation_version = "v1:model"
    monkeypatch.setattr("radius_rag.observability.event.load_retrieval_config", lambda: cfg)

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "radius_rag.observability.event.emit",
        lambda event: emitted.append(event) or True,
    )
    rag_service = MagicMock(metadata=[{"file": "a.py"}])
    maybe_emit_retrieval_event(
        rag_service=rag_service,
        query="hello",
        duration_ms=12,
        results=[{"metadata_index": 0, "code": "x", "score": 1.0}],
        trace_kwargs={
            "faiss_ranks": {0: 1},
            "bm25_ranks": {0: 1},
            "faiss_scores": {0: 0.5},
            "bm25_scores": {0: 0.5},
            "candidates": [{"metadata_index": 0, "code": "x", "score": 1.0}],
        },
    )

    retrieve_mock.assert_not_called()
    assert len(emitted) == 1
    assert emitted[0]["_pending_snapshot"] is True
    assert emitted[0]["snapshot_input"]["query"] == "hello"
