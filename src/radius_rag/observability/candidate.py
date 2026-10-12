"""Normalized retrieval candidate for judge input and training exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalCandidate:
    metadata_index: int | None
    stable_id: str | None
    rank: int | None
    score: float | None
    text: str
    text_present: bool
    text_source_field: str | None
    code_hash: str | None
    document_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_judge_dict(self) -> dict[str, Any]:
        return {
            "metadata_index": self.metadata_index,
            "stable_id": self.stable_id,
            "rank": self.rank,
            "score": self.score,
            "code": self.text,
            "file": self.document_metadata.get("file", ""),
            "repo": self.document_metadata.get("repo", ""),
        }


_TEXT_FIELDS = ("code", "text", "content", "chunk_text", "body")


def extract_chunk_text(item: dict[str, Any]) -> tuple[str, bool, str | None]:
    """Return (text, present, source_field). Empty string if no text field."""
    for key in _TEXT_FIELDS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value, True, key
    chunk_value = item.get("chunk_value")
    if isinstance(chunk_value, str) and chunk_value.strip():
        return chunk_value, True, "chunk_value"
    return "", False, None


def normalize_candidate(
    item: dict[str, Any],
    *,
    rank: int | None = None,
    stable_id: str | None = None,
    code_hash: str | None = None,
    document_metadata: dict[str, Any] | None = None,
) -> RetrievalCandidate:
    text, present, source = extract_chunk_text(item)
    meta = document_metadata if document_metadata is not None else {}
    if not meta:
        meta = {
            key: item.get(key)
            for key in (
                "repo",
                "file",
                "language",
                "source_type",
                "name",
                "full_name",
                "detail",
                "start_line",
                "end_line",
                "git_url",
                "updated_at",
                "doc_hash",
                "chunk_value",
            )
            if key in item
        }
    return RetrievalCandidate(
        metadata_index=item.get("metadata_index") if isinstance(item.get("metadata_index"), int) else None,
        stable_id=stable_id or item.get("stable_id"),
        rank=rank if rank is not None else item.get("top_k_rank") or item.get("rank"),
        score=float(item["score"]) if item.get("score") is not None else None,
        text=text,
        text_present=present,
        text_source_field=source,
        code_hash=code_hash or item.get("code_hash"),
        document_metadata=meta,
        raw=dict(item),
    )


def candidates_from_trace(trace: dict[str, Any]) -> list[RetrievalCandidate]:
    chunk_texts = trace.get("chunk_texts")
    by_index: dict[int, dict[str, Any]] = {}
    if isinstance(chunk_texts, list):
        for entry in chunk_texts:
            if isinstance(entry, dict) and isinstance(entry.get("metadata_index"), int):
                by_index[entry["metadata_index"]] = entry

    rows = trace.get("candidates")
    if not isinstance(rows, list):
        rows = trace.get("top_k") or []

    result: list[RetrievalCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("metadata_index")
        merged = dict(row)
        if isinstance(idx, int) and idx in by_index:
            extra = by_index[idx]
            if merged.get("code") in (None, "") and extra.get("code"):
                merged["code"] = extra["code"]
            if not merged.get("code_hash"):
                merged["code_hash"] = extra.get("code_hash")
            if not merged.get("document_metadata"):
                merged["document_metadata"] = extra.get("document_metadata")
        in_top = bool(row.get("in_top_k"))
        rank = row.get("top_k_rank") if in_top else None
        result.append(
            normalize_candidate(
                merged,
                rank=rank if isinstance(rank, int) else None,
                stable_id=merged.get("stable_id"),
                code_hash=merged.get("code_hash"),
                document_metadata=merged.get("document_metadata"),
            )
        )
    result.sort(
        key=lambda c: (
            c.rank if c.rank is not None else 10_000,
            -(c.score or 0.0),
        )
    )
    return result


def candidates_from_response_payload(payload: dict[str, Any]) -> list[RetrievalCandidate]:
    results = payload.get("results")
    if isinstance(results, list) and results:
        return [normalize_candidate(item, rank=index + 1) for index, item in enumerate(results) if isinstance(item, dict)]

    sources = payload.get("sources")
    if isinstance(sources, list) and sources:
        return [normalize_candidate(item, rank=index + 1) for index, item in enumerate(sources) if isinstance(item, dict)]

    return []
