from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from radius_rag.observability.evaluation.judge import (
    resolve_judge_candidate_sets,
    validate_judge_response,
)
from radius_rag.observability.snapshot import finalize_pending_event, prepare_frozen_snapshot_input


def _valid_scores(chunk_count: int) -> dict:
    return {
        "per_chunk_scores": [
            {"chunk_index": i, "relevance": 2, "rationale": "ok"} for i in range(chunk_count)
        ],
        "context_sufficiency": {"sufficient": True, "missing_topics": []},
        "query_classification": {"category": "answerable"},
        "context_sufficiency_score": 2,
        "explanation": "ok",
    }


def test_validate_judge_response_requires_exact_index_set() -> None:
    payload = _valid_scores(2)
    payload["per_chunk_scores"] = payload["per_chunk_scores"][:1]
    with pytest.raises(ValueError, match="exactly once"):
        validate_judge_response(payload, 2)


def test_validate_judge_response_rejects_duplicate_chunk_index() -> None:
    payload = _valid_scores(1)
    payload["per_chunk_scores"] = [
        {"chunk_index": 0, "relevance": 2},
        {"chunk_index": 0, "relevance": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_judge_response(payload, 1)


def test_resolve_judge_candidate_sets_uses_response_as_final_top_k() -> None:
    event = {
        "response_payload": {
            "results": [{"code": "client", "metadata_index": 1, "score": 0.9}],
        },
        "retrieval_trace": {
            "candidates": [
                {"metadata_index": 0, "score": 0.1, "in_top_k": False},
                {"metadata_index": 1, "score": 0.9, "in_top_k": True, "top_k_rank": 1, "code": "trace"},
            ],
        },
    }
    final_top_k, pool, diag = resolve_judge_candidate_sets(event)
    assert len(final_top_k) == 1
    assert final_top_k[0].text == "client"
    assert len(pool) == 2
    assert diag["final_top_k_size"] == 1


def test_finalize_pending_event_uses_frozen_index_info() -> None:
    pending = {
        "event_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_input": {
            "query": "q",
            "results": [],
            "trace_kwargs": {},
            "request_payload": {"query": "q"},
            "retrieval_config_snapshot": {},
            "embedding_model": "emb",
            "index_info": {
                "staging_id": "staging-frozen",
                "index_fingerprint": "fingerprint-frozen",
                "embedding_model": "emb-frozen",
                "source": "local",
                "document_count": 10,
                "dimensions": 384,
                "manifest": {"built_at": "2020-01-01"},
            },
        },
    }
    row = finalize_pending_event(pending)
    assert row["index_staging_id"] == "staging-frozen"
    assert row["index_fingerprint"] == "fingerprint-frozen"
    assert row["embedding_model"] == "emb-frozen"


def test_prepare_frozen_snapshot_captures_metadata_at_emit_time() -> None:
    rag = MagicMock()
    rag.metadata = [{"code": "frozen-body", "file": "a.py"}]
    with patch(
        "radius_rag.observability.snapshot.get_active_index_info",
        return_value=MagicMock(
            staging_id="s1",
            index_fingerprint="fp",
            embedding_model="e",
            source="local",
            document_count=1,
            dimensions=8,
            manifest={},
        ),
    ):
        frozen = prepare_frozen_snapshot_input(
            rag_service=rag,
            query="q",
            results=[{"metadata_index": 0, "score": 1.0}],
            trace_kwargs={"candidates": [{"metadata_index": 0, "score": 1.0}]},
            request_payload={"query": "q"},
            retrieval_config_snapshot={},
            embedding_model="e",
        )
    enriched = frozen["trace_kwargs"]["candidates"][0]
    assert enriched.get("code") == "frozen-body"
    assert frozen["index_info"]["index_fingerprint"] == "fp"


def test_worker_uses_atomic_commit() -> None:
    async def _run() -> None:
        from radius_rag.observability.config import MIGRATIONS_DIR, RetrievalConfig
        from radius_rag.observability.evaluation.worker import _process_job

        cfg = RetrievalConfig(
            telemetry_enabled=False,
            evaluation_auto_enqueue=False,
            database_url="postgresql://example",
            queue_size=10,
            batch_size=10,
            flush_interval_seconds=1.0,
            evaluator_concurrency=1,
            lease_ttl_seconds=60,
            max_retries=3,
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
            compression_config_key="k",
            judge_relevance_threshold=2,
            judge_max_query_chars=1000,
            judge_max_chunk_chars=1000,
            judge_max_chunks=5,
            judge_json_max_retries=1,
            migrations_dir=MIGRATIONS_DIR,
        )
        token = uuid4()
        job = {
            "job_id": 1,
            "event_id": uuid4(),
            "event_created_at": datetime.now(timezone.utc),
            "attempts": 1,
            "max_attempts": 4,
            "lease_token": token,
        }
        commit = AsyncMock(return_value=True)
        finish = AsyncMock(return_value=True)
        with patch(
            "radius_rag.observability.evaluation.worker.mark_evaluation_job_running",
            AsyncMock(return_value=True),
        ):
            with patch(
                "radius_rag.observability.evaluation.worker.fetch_retrieval_event",
                AsyncMock(return_value={"response_payload": {"sources": [{"code": "x"}]}}),
            ):
                with patch(
                    "radius_rag.observability.evaluation.worker.run_judge_for_event",
                    AsyncMock(
                        return_value={
                            "status": "completed",
                            "judge_model": "j",
                            "prompt_version": "v1",
                            "per_chunk_scores": [],
                            "context_sufficiency": {},
                            "query_classification": {},
                            "aggregated_metrics": {},
                            "explanation": "",
                            "diagnostics": {},
                            "truncation_info": {},
                        }
                    ),
                ):
                    with patch(
                        "radius_rag.observability.evaluation.worker.commit_evaluation_and_complete_job",
                        commit,
                    ):
                        with patch(
                            "radius_rag.observability.evaluation.worker.finish_evaluation_job",
                            finish,
                        ):
                            await _process_job(job, cfg, AsyncMock())
                            commit.assert_awaited_once()
                            finish.assert_not_awaited()

    asyncio.run(_run())


def test_claim_evaluation_jobs_sql_excludes_exhausted_attempts() -> None:
    import inspect

    from radius_rag.observability.storage import db as db_mod

    source = inspect.getsource(db_mod.claim_evaluation_jobs)
    assert "attempts < max_attempts" in source
    assert "status = 'dead'" in source
