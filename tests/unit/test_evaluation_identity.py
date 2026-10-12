from __future__ import annotations

from radius_rag.observability.evaluation.identity import (
    EvaluationIdentity,
    build_evaluation_version,
)


def test_build_evaluation_version() -> None:
    assert build_evaluation_version("v1", "qwen-judge") == "v1:qwen-judge"


def test_parse_roundtrip() -> None:
    ident = EvaluationIdentity.parse("v2:model-x")
    assert ident.prompt_version == "v2"
    assert ident.judge_model == "model-x"
    assert ident.evaluation_version == "v2:model-x"
