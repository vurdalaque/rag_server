from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4


def test_worker_retries_with_next_attempt_at() -> None:
    async def _run() -> None:
        from retrieval_config import MIGRATIONS_DIR, RetrievalConfig
        from retrieval_eval_worker import _process_job

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

        with patch(
            "radius_rag.observability.evaluation.worker.mark_evaluation_job_running",
            AsyncMock(return_value=True),
        ):
            with patch(
                "radius_rag.observability.evaluation.worker.fetch_retrieval_event",
                AsyncMock(return_value={"response_payload": {"results": []}}),
            ):
                with patch(
                    "radius_rag.observability.evaluation.worker.run_judge_for_event",
                    AsyncMock(side_effect=RuntimeError("llm down")),
                ):
                    finish = AsyncMock(return_value=True)
                    with patch(
                        "radius_rag.observability.evaluation.worker.finish_evaluation_job",
                        finish,
                    ):
                        await _process_job(job, cfg, AsyncMock())
                        finish.assert_awaited()
                        assert finish.await_args.kwargs["status"] == "pending"
                        assert finish.await_args.kwargs["next_attempt_at"] is not None

    asyncio.run(_run())
