from __future__ import annotations

from radius_rag.observability.evaluation.job_errors import (
    classify_job_failure,
    job_error_retryable,
)


def test_empty_final_top_k_not_retryable() -> None:
    err = "{'reason': 'empty_final_top_k', 'final_top_k_size': 0}"
    assert classify_job_failure(err) == "empty_final_top_k"
    assert job_error_retryable(err) is False


def test_judge_error_retryable() -> None:
    err = "per_chunk_scores must be a list"
    assert classify_job_failure(err) == "judge_validation"
    assert job_error_retryable(err) is True
