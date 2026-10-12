from __future__ import annotations

from pathlib import Path

from radius_rag.observability.config import RetrievalConfig
from radius_rag.observability.evaluation.identity import (
    EvaluationIdentity,
    build_evaluation_version,
    evaluation_version_from_config,
    methodology_fingerprint_from_config,
)


def _config(**overrides: object) -> RetrievalConfig:
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
        "judge_llm_model": "qwen-judge",
        "judge_llm_temperature": 0.0,
        "judge_llm_top_p": 1.0,
        "judge_llm_top_k": 20,
        "judge_llm_presence_penalty": 0.0,
        "judge_llm_repetition_penalty": 1.0,
        "judge_llm_enable_thinking": False,
        "judge_llm_timeout_seconds": 30.0,
        "compression_config_key": "k",
        "judge_relevance_threshold": 2,
        "judge_max_query_chars": 8000,
        "judge_max_chunk_chars": 6000,
        "judge_max_chunks": 30,
        "judge_json_max_retries": 2,
        "migrations_dir": Path("."),
    }
    base.update(overrides)
    return RetrievalConfig(**base)


def test_build_evaluation_version_legacy_without_fingerprint() -> None:
    assert build_evaluation_version("v1", "qwen-judge") == "v1:qwen-judge"


def test_parse_roundtrip_with_fingerprint() -> None:
    version = "v2:model-x+abc123def4567890"
    ident = EvaluationIdentity.parse(version)
    assert ident.prompt_version == "v2"
    assert ident.judge_model == "model-x"
    assert ident.methodology_fingerprint == "abc123def4567890"
    assert ident.evaluation_version == version


def test_methodology_fingerprint_stable_across_restarts() -> None:
    cfg_a = _config()
    cfg_b = _config()
    assert methodology_fingerprint_from_config(cfg_a) == methodology_fingerprint_from_config(cfg_b)


def test_methodology_change_changes_evaluation_version() -> None:
    base = evaluation_version_from_config(_config())
    changed = evaluation_version_from_config(_config(judge_relevance_threshold=3))
    assert base != changed
    assert "+" in changed


def test_operational_timeout_does_not_change_evaluation_version() -> None:
    a = evaluation_version_from_config(_config(judge_llm_timeout_seconds=30.0))
    b = evaluation_version_from_config(_config(judge_llm_timeout_seconds=120.0))
    assert a == b
