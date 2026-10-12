from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from retrieval_config import RetrievalConfig
from retrieval_judge import (
    _parse_judge_json,
    run_judge_for_event,
    validate_judge_response,
)


def _test_config(**overrides: Any) -> RetrievalConfig:
    base = {
        "telemetry_enabled": False,
        "evaluation_auto_enqueue": False,
        "database_url": "",
        "queue_size": 10,
        "batch_size": 10,
        "flush_interval_seconds": 1.0,
        "evaluator_concurrency": 1,
        "lease_ttl_seconds": 30,
        "max_retries": 1,
        "retry_base_seconds": 1.0,
        "poll_interval_seconds": 1.0,
        "judge_prompt_version": "v1",
        "judge_llm_url": "http://127.0.0.1:8001/v1/chat/completions",
        "judge_llm_model": "judge",
        "judge_llm_temperature": 0.0,
        "judge_llm_top_p": 1.0,
        "judge_llm_top_k": 20,
        "judge_llm_presence_penalty": 0.0,
        "judge_llm_repetition_penalty": 1.0,
        "judge_llm_enable_thinking": False,
        "judge_llm_timeout_seconds": 30.0,
        "compression_config_key": "key",
        "judge_relevance_threshold": 2,
        "judge_max_query_chars": 8000,
        "judge_max_chunk_chars": 6000,
        "judge_max_chunks": 30,
        "judge_json_max_retries": 2,
        "migrations_dir": __import__("pathlib").Path("."),
    }
    base.update(overrides)
    return RetrievalConfig(**base)


def _valid_judge_payload(chunk_count: int = 1) -> dict[str, Any]:
    return {
        "per_chunk_scores": [
            {"chunk_index": index, "relevance": 2, "rationale": "ok"}
            for index in range(chunk_count)
        ],
        "context_sufficiency": {"sufficient": True, "missing_topics": []},
        "query_classification": {"category": "answerable"},
        "context_sufficiency_score": 2,
        "explanation": "fine",
    }


def test_parse_judge_json_strips_markdown_fence() -> None:
    payload = _valid_judge_payload()
    raw = "```json\n" + json.dumps(payload) + "\n```"
    parsed = _parse_judge_json(raw)
    assert parsed["explanation"] == "fine"


def test_validate_judge_response_rejects_bad_chunk_index() -> None:
    payload = _valid_judge_payload(1)
    payload["per_chunk_scores"][0]["chunk_index"] = 5
    with pytest.raises(ValueError, match="chunk_index"):
        validate_judge_response(payload, 1)


def test_run_judge_retries_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _test_config(judge_json_max_retries=2)
    calls: list[int] = []

    async def fake_chat(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "not-json"
        return json.dumps(_valid_judge_payload())

    monkeypatch.setattr(
        "radius_rag.observability.evaluation.judge.judge_multimodal_chat",
        fake_chat,
    )

    event = {
        "request_payload": {"query": "q"},
        "response_payload": {"sources": [{"code": "chunk", "file": "a.py"}]},
    }
    result = asyncio.run(run_judge_for_event(event, cfg))
    assert result["status"] == "completed"
    assert len(calls) == 2


def test_run_judge_retries_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _test_config(judge_json_max_retries=2)
    calls = 0

    async def fake_chat(*_args: Any, **_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            bad = _valid_judge_payload(1)
            bad["per_chunk_scores"] = []
            return json.dumps(bad)
        return json.dumps(_valid_judge_payload(1))

    monkeypatch.setattr(
        "radius_rag.observability.evaluation.judge.judge_multimodal_chat",
        fake_chat,
    )

    event = {
        "request_payload": {"query": "q"},
        "response_payload": {"sources": [{"code": "chunk", "file": "a.py"}]},
    }
    result = asyncio.run(run_judge_for_event(event, cfg))
    assert result["status"] == "completed"
    assert calls == 2
