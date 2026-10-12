"""Historical retrieval data export for training pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Literal
from uuid import UUID

from radius_rag.observability.analytics.analytics import EventFilters, JudgeVersionFilter, _event_where, _row_to_json
from radius_rag.observability.config import RetrievalConfig, load_retrieval_config
from radius_rag.observability.storage.db import get_pool

ExportFormat = Literal["traces", "pairs", "triples", "labeled"]
EXPORT_VERSION = "1"


@dataclass(frozen=True)
class ExportCriteria:
    filters: EventFilters
    judge: JudgeVersionFilter
    min_relevance: int = 2
    require_context_sufficient: bool = False


def build_export_manifest(criteria: ExportCriteria, export_format: ExportFormat) -> dict[str, Any]:
    cfg = load_retrieval_config()
    return {
        "export_version": EXPORT_VERSION,
        "format": export_format,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "criteria": {
            "start": criteria.filters.start.isoformat(),
            "end": criteria.filters.end.isoformat(),
            "index_fingerprint": criteria.filters.index_fingerprint,
            "embedding_model": criteria.filters.embedding_model,
            "index_staging_id": criteria.filters.index_staging_id,
            "judge_prompt_version": criteria.judge.prompt_version,
            "judge_model": criteria.judge.judge_model,
            "min_relevance": criteria.min_relevance,
            "require_context_sufficient": criteria.require_context_sufficient,
            "judge_relevance_threshold": cfg.judge_relevance_threshold,
        },
    }


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _event_query_text(event: dict[str, Any]) -> str:
    request = _parse_json_field(event.get("request_payload")) or {}
    if isinstance(request.get("query"), str):
        return request["query"]
    trace = _parse_json_field(event.get("retrieval_trace")) or {}
    query = trace.get("query")
    return query if isinstance(query, str) else ""


def _chunk_entries(event: dict[str, Any]) -> list[dict[str, Any]]:
    trace = _parse_json_field(event.get("retrieval_trace")) or {}
    texts = trace.get("chunk_texts")
    if isinstance(texts, list) and texts:
        return [item for item in texts if isinstance(item, dict)]
    response = _parse_json_field(event.get("response_payload")) or {}
    results = response.get("results") or response.get("sources") or []
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def _score_map(evaluation: dict[str, Any]) -> dict[int, int]:
    scores: dict[int, int] = {}
    per_chunk = _parse_json_field(evaluation.get("per_chunk_scores")) or []
    if not isinstance(per_chunk, list):
        return scores
    for entry in per_chunk:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("chunk_index")
        rel = entry.get("relevance")
        if isinstance(idx, int) and isinstance(rel, int):
            scores[idx] = rel
    return scores


async def _iter_evaluated_events(
    criteria: ExportCriteria,
    *,
    config: RetrievalConfig | None = None,
) -> AsyncIterator[tuple[dict[str, Any], dict[str, Any]]]:
    where, params = _event_where(criteria.filters)
    eval_clauses = ["ev.status = 'completed'"]
    if criteria.judge.prompt_version:
        params.append(criteria.judge.prompt_version)
        eval_clauses.append(f"ev.prompt_version = ${len(params)}")
    if criteria.judge.judge_model:
        params.append(criteria.judge.judge_model)
        eval_clauses.append(f"ev.judge_model = ${len(params)}")
    if criteria.require_context_sufficient:
        eval_clauses.append("(ev.context_sufficiency->>'sufficient')::boolean IS TRUE")

    pool = await get_pool(config)
    rows = await pool.fetch(
        f"""
        SELECT
            e.*,
            ev.evaluation_id,
            ev.evaluated_at,
            ev.judge_model,
            ev.prompt_version,
            ev.per_chunk_scores,
            ev.context_sufficiency,
            ev.query_classification,
            ev.aggregated_metrics,
            ev.explanation
        FROM retrieval_obs.retrieval_events e
        INNER JOIN retrieval_obs.retrieval_evaluations ev
            ON e.event_id = ev.event_id AND e.created_at = ev.event_created_at
        WHERE {where} AND {" AND ".join(eval_clauses)}
        ORDER BY e.created_at, e.event_id
        """,
        *params,
    )
    for row in rows:
        event = {key: row[key] for key in row.keys() if not key.startswith("ev_")}
        evaluation = {
            "evaluation_id": row["evaluation_id"],
            "evaluated_at": row["evaluated_at"],
            "judge_model": row["judge_model"],
            "prompt_version": row["prompt_version"],
            "per_chunk_scores": row["per_chunk_scores"],
            "context_sufficiency": row["context_sufficiency"],
            "query_classification": row["query_classification"],
            "aggregated_metrics": row["aggregated_metrics"],
            "explanation": row["explanation"],
        }
        yield _row_to_json(event), _row_to_json(evaluation)


async def export_jsonl(
    export_format: ExportFormat,
    criteria: ExportCriteria,
    *,
    config: RetrievalConfig | None = None,
) -> tuple[dict[str, Any], AsyncIterator[str]]:
    manifest = build_export_manifest(criteria, export_format)

    async def _lines() -> AsyncIterator[str]:
        if export_format == "traces":
            where, params = _event_where(criteria.filters)
            pool = await get_pool(config)
            rows = await pool.fetch(
                f"""
                SELECT *
                FROM retrieval_obs.retrieval_events e
                WHERE {where}
                ORDER BY e.created_at, e.event_id
                """,
                *params,
            )
            for row in rows:
                yield json.dumps(_row_to_json(row), ensure_ascii=False, default=str) + "\n"
            return

        async for event, evaluation in _iter_evaluated_events(criteria, config=config):
            query = _event_query_text(event)
            chunks = _chunk_entries(event)
            scores = _score_map(evaluation)
            positives = []
            negatives = []
            labeled = []
            for index, chunk in enumerate(chunks):
                relevance = scores.get(index, scores.get(chunk.get("metadata_index", -1), 0))
                entry = {
                    "query": query,
                    "chunk": chunk,
                    "relevance": relevance,
                    "event_id": event.get("event_id"),
                    "index_fingerprint": event.get("index_fingerprint"),
                    "embedding_model": event.get("embedding_model"),
                    "judge_model": evaluation.get("judge_model"),
                    "prompt_version": evaluation.get("prompt_version"),
                }
                labeled.append(entry)
                if relevance >= criteria.min_relevance:
                    positives.append(chunk)
                elif relevance == 0:
                    negatives.append(chunk)

            if export_format == "labeled":
                for item in labeled:
                    yield json.dumps(item, ensure_ascii=False, default=str) + "\n"
            elif export_format == "pairs":
                for chunk in positives:
                    yield (
                        json.dumps(
                            {
                                "query": query,
                                "positive": chunk,
                                "event_id": event.get("event_id"),
                                "relevance_threshold": criteria.min_relevance,
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
            elif export_format == "triples":
                if not positives:
                    continue
                positive = positives[0]
                negative = negatives[0] if negatives else None
                if negative is None:
                    continue
                yield (
                    json.dumps(
                        {
                            "query": query,
                            "positive": positive,
                            "negative": negative,
                            "event_id": event.get("event_id"),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

    return manifest, _lines()
