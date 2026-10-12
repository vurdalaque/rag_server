"""Evaluation identity shared by jobs, results, reconciliation, and analytics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radius_rag.observability.config import RetrievalConfig

METHODOLOGY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationIdentity:
    prompt_version: str
    judge_model: str
    methodology_fingerprint: str | None = None

    @property
    def evaluation_version(self) -> str:
        return build_evaluation_version(
            self.prompt_version,
            self.judge_model,
            self.methodology_fingerprint,
        )

    @classmethod
    def parse(cls, evaluation_version: str) -> EvaluationIdentity:
        methodology_fingerprint: str | None = None
        base = evaluation_version
        if "+" in evaluation_version:
            base, methodology_fingerprint = evaluation_version.rsplit("+", 1)
        if ":" in base:
            prompt, model = base.split(":", 1)
            return cls(
                prompt_version=prompt,
                judge_model=model,
                methodology_fingerprint=methodology_fingerprint,
            )
        return cls(
            prompt_version=base,
            judge_model="",
            methodology_fingerprint=methodology_fingerprint,
        )


def methodology_payload_from_config(config: RetrievalConfig) -> dict[str, Any]:
    """Canonical judge methodology fields that affect scoring semantics."""
    return {
        "schema": METHODOLOGY_SCHEMA_VERSION,
        "prompt_version": config.judge_prompt_version,
        "judge_model": config.judge_llm_model,
        "relevance_threshold": config.judge_relevance_threshold,
        "max_query_chars": config.judge_max_query_chars,
        "max_chunk_chars": config.judge_max_chunk_chars,
        "max_chunks": config.judge_max_chunks,
        "llm_temperature": config.judge_llm_temperature,
        "llm_top_p": config.judge_llm_top_p,
        "llm_top_k": config.judge_llm_top_k,
        "llm_presence_penalty": config.judge_llm_presence_penalty,
        "llm_repetition_penalty": config.judge_llm_repetition_penalty,
        "llm_enable_thinking": config.judge_llm_enable_thinking,
    }


def methodology_fingerprint_from_config(config: RetrievalConfig) -> str:
    payload = methodology_payload_from_config(config)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def evaluation_version_from_config(config: RetrievalConfig) -> str:
    return build_evaluation_version(
        config.judge_prompt_version,
        config.judge_llm_model,
        methodology_fingerprint_from_config(config),
    )


def build_evaluation_version(
    prompt_version: str,
    judge_model: str,
    methodology_fingerprint: str | None = None,
) -> str:
    base = f"{prompt_version}:{judge_model}"
    if methodology_fingerprint:
        return f"{base}+{methodology_fingerprint}"
    return base
