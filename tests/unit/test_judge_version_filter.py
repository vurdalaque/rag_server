from __future__ import annotations

from radius_rag.observability.analytics.analytics import JudgeVersionFilter


def test_judge_version_shorthand_filters_prompt_and_model() -> None:
    judge = JudgeVersionFilter.parse("v1:qwen-judge")
    assert judge.raw_evaluation_version is None
    assert judge.prompt_version == "v1"
    assert judge.judge_model == "qwen-judge"
    assert judge.evaluation_version is None


def test_judge_version_with_fingerprint_is_exact_match() -> None:
    version = "v1:qwen-judge+abc123def4567890"
    judge = JudgeVersionFilter.parse(version)
    assert judge.raw_evaluation_version == version
    assert judge.evaluation_version == version
