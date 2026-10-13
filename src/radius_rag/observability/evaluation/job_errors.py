"""Classify evaluation job failures for operator tooling."""

from __future__ import annotations

_NON_RETRYABLE_MARKERS = (
    "empty_final_top_k",
    "missing_chunk_text",
)


def classify_job_failure(last_error: str | None) -> str | None:
    if not last_error:
        return None
    lowered = last_error.lower()
    for marker in _NON_RETRYABLE_MARKERS:
        if marker in lowered:
            return marker
    if "per_chunk_scores" in lowered:
        return "judge_validation"
    if "context_sufficiency" in lowered:
        return "judge_validation"
    return "other"


def job_error_retryable(last_error: str | None) -> bool:
    reason = classify_job_failure(last_error)
    return reason not in _NON_RETRYABLE_MARKERS
