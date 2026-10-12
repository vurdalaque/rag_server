from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from retrieval_config import MIGRATIONS_DIR, RetrievalConfig


def test_flush_persists_events_when_enqueue_fails() -> None:
    async def _run() -> None:
        from retrieval_telemetry import _flush_batch

        cfg = RetrievalConfig(
            telemetry_enabled=True,
            evaluation_auto_enqueue=True,
            database_url="postgresql://example",
            queue_size=10,
            batch_size=10,
            flush_interval_seconds=1.0,
            evaluator_concurrency=1,
            lease_ttl_seconds=30,
            max_retries=2,
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

        event = {
            "event_id": "00000000-0000-4000-8000-000000000001",
            "created_at": "2026-01-01T00:00:00+00:00",
            "request_payload": {},
            "response_payload": {},
            "retrieval_trace": {},
            "metadata": {},
        }

        with patch("radius_rag.observability.telemetry.writer._config", cfg):
            with patch(
                "radius_rag.observability.telemetry.writer.insert_events",
                AsyncMock(return_value=1),
            ) as insert:
                with patch(
                    "radius_rag.observability.telemetry.writer._enqueue_jobs_for_events",
                    AsyncMock(side_effect=RuntimeError("enqueue down")),
                ):
                    await _flush_batch([event])
                    insert.assert_awaited_once()

    asyncio.run(_run())
