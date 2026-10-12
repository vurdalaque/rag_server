"""Evaluation identity shared by jobs, results, reconciliation, and analytics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationIdentity:
    prompt_version: str
    judge_model: str

    @property
    def evaluation_version(self) -> str:
        return build_evaluation_version(self.prompt_version, self.judge_model)

    @classmethod
    def parse(cls, evaluation_version: str) -> EvaluationIdentity:
        if ":" in evaluation_version:
            prompt, model = evaluation_version.split(":", 1)
            return cls(prompt_version=prompt, judge_model=model)
        return cls(prompt_version=evaluation_version, judge_model="")


def build_evaluation_version(prompt_version: str, judge_model: str) -> str:
    return f"{prompt_version}:{judge_model}"
