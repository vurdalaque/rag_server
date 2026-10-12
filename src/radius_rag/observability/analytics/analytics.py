"""SQL-backed analytics for retrieval observability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping
from uuid import UUID

from radius_rag.observability.config import RetrievalConfig, load_retrieval_config
from radius_rag.observability.storage.db import get_pool

BucketGranularity = Literal["hour", "day", "week"]

_BUCKET_TRUNC = {
    "hour": "hour",
    "day": "day",
    "week": "week",
}


@dataclass(frozen=True)
class EventFilters:
    start: datetime
    end: datetime
    index_fingerprint: str | None = None
    embedding_model: str | None = None
    index_staging_id: str | None = None
    request_source: str | None = None


@dataclass(frozen=True)
class JudgeVersionFilter:
    prompt_version: str | None = None
    judge_model: str | None = None
    methodology_fingerprint: str | None = None
    raw_evaluation_version: str | None = None

    @classmethod
    def parse(cls, evaluation_version: str | None) -> JudgeVersionFilter:
        if not evaluation_version:
            return cls()
        from radius_rag.observability.evaluation.identity import EvaluationIdentity

        stripped = evaluation_version.strip()
        ident = EvaluationIdentity.parse(stripped)
        # Exact evaluation_version match only when a methodology fingerprint is present
        # (e.g. v1:qwen-judge+a1b2c3d4). Shorthand v1:model filters prompt + judge_model.
        raw = stripped if ident.methodology_fingerprint else None
        return cls(
            prompt_version=ident.prompt_version or None,
            judge_model=ident.judge_model or None,
            methodology_fingerprint=ident.methodology_fingerprint,
            raw_evaluation_version=raw,
        )

    @property
    def evaluation_version(self) -> str | None:
        """Full evaluation_version string for exact SQL match (includes +fingerprint when set)."""
        return self.raw_evaluation_version


def _append_filter(
    clauses: list[str],
    params: list[Any],
    column: str,
    value: str | None,
) -> None:
    if value is None:
        return
    params.append(value)
    clauses.append(f"{column} = ${len(params)}")


def _event_where(filters: EventFilters, alias: str = "e") -> tuple[str, list[Any]]:
    clauses = [
        f"{alias}.created_at >= $1",
        f"{alias}.created_at < $2",
    ]
    params: list[Any] = [filters.start, filters.end]
    _append_filter(clauses, params, f"{alias}.index_fingerprint", filters.index_fingerprint)
    _append_filter(clauses, params, f"{alias}.embedding_model", filters.embedding_model)
    _append_filter(clauses, params, f"{alias}.index_staging_id", filters.index_staging_id)
    _append_filter(clauses, params, f"{alias}.request_source", filters.request_source)
    return " AND ".join(clauses), params


def _eval_where(
    filters: EventFilters,
    judge: JudgeVersionFilter,
    alias: str = "ev",
) -> tuple[str, list[Any]]:
    where, params = _event_where(filters, alias="e")
    join = (
        " INNER JOIN retrieval_obs.retrieval_events e "
        f"ON e.event_id = {alias}.event_id AND e.created_at = {alias}.event_created_at "
    )
    clauses = [where, f"{alias}.status = 'completed'"]
    if judge.evaluation_version:
        _append_filter(clauses, params, f"{alias}.evaluation_version", judge.evaluation_version)
    else:
        _append_filter(clauses, params, f"{alias}.prompt_version", judge.prompt_version)
        _append_filter(clauses, params, f"{alias}.judge_model", judge.judge_model)
    return join, " AND ".join(clauses), params


def _row_to_json(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _serialize_value(row[key]) for key in row.keys()}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


async def fetch_period_event_stats(
    filters: EventFilters,
    bucket: BucketGranularity,
    *,
    config: RetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    trunc = _BUCKET_TRUNC[bucket]
    where, params = _event_where(filters)
    pool = await get_pool(config)
    rows = await pool.fetch(
        f"""
        SELECT
            date_trunc('{trunc}', created_at) AS bucket,
            COUNT(*)::bigint AS event_count,
            COUNT(*) FILTER (WHERE status = 'ok')::bigint AS ok_count,
            COUNT(*) FILTER (WHERE status = 'error')::bigint AS error_count,
            COUNT(*) FILTER (
                WHERE COALESCE(
                    (response_payload->>'count')::int,
                    jsonb_array_length(COALESCE(response_payload->'results', '[]'::jsonb))
                ) = 0
            )::bigint AS zero_result_count,
            AVG(duration_ms)::float AS avg_duration_ms
        FROM retrieval_obs.retrieval_events e
        WHERE {where}
        GROUP BY 1
        ORDER BY 1
        """,
        *params,
    )
    return [_row_to_json(row) for row in rows]


async def fetch_latency_percentiles(
    filters: EventFilters,
    *,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    where, params = _event_where(filters)
    pool = await get_pool(config)
    row = await pool.fetchrow(
        f"""
        SELECT
            percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_ms,
            COUNT(*) FILTER (WHERE duration_ms IS NOT NULL)::bigint AS sample_count
        FROM retrieval_obs.retrieval_events e
        WHERE {where} AND duration_ms IS NOT NULL
        """,
        *params,
    )
    return _row_to_json(row) if row else {}


async def fetch_coverage(
    filters: EventFilters,
    judge: JudgeVersionFilter,
    *,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_retrieval_config()
    where, params = _event_where(filters)
    pool = await get_pool(cfg)

    total = await pool.fetchval(
        f"SELECT COUNT(*)::bigint FROM retrieval_obs.retrieval_events e WHERE {where}",
        *params,
    )

    eval_clauses = [
        "ev.status = 'completed'",
        "e.created_at >= $1",
        "e.created_at < $2",
    ]
    eval_params: list[Any] = [filters.start, filters.end]
    _append_filter(eval_clauses, eval_params, "e.index_fingerprint", filters.index_fingerprint)
    _append_filter(eval_clauses, eval_params, "e.embedding_model", filters.embedding_model)
    _append_filter(eval_clauses, eval_params, "e.index_staging_id", filters.index_staging_id)
    _append_filter(eval_clauses, eval_params, "e.request_source", filters.request_source)
    if judge.evaluation_version:
        _append_filter(eval_clauses, eval_params, "ev.evaluation_version", judge.evaluation_version)
    else:
        _append_filter(eval_clauses, eval_params, "ev.prompt_version", judge.prompt_version)
        _append_filter(eval_clauses, eval_params, "ev.judge_model", judge.judge_model)

    evaluated = await pool.fetchval(
        f"""
        SELECT COUNT(DISTINCT (ev.event_id, ev.event_created_at))::bigint
        FROM retrieval_obs.retrieval_evaluations ev
        INNER JOIN retrieval_obs.retrieval_events e
            ON e.event_id = ev.event_id AND e.created_at = ev.event_created_at
        WHERE {" AND ".join(eval_clauses)}
        """,
        *eval_params,
    )

    evaluation_version = judge.raw_evaluation_version or cfg.evaluation_version

    pending = await pool.fetchval(
        """
        SELECT COUNT(*)::bigint
        FROM retrieval_obs.evaluation_jobs
        WHERE evaluation_version = $1
          AND status IN ('pending', 'leased', 'running')
        """,
        evaluation_version,
    )

    return {
        "total_events": int(total or 0),
        "evaluated_events": int(evaluated or 0),
        "pending_jobs": int(pending or 0),
        "coverage_ratio": (int(evaluated or 0) / int(total)) if total else 0.0,
        "judge_relevance_threshold": cfg.judge_relevance_threshold,
        "evaluation_version": evaluation_version,
    }


async def fetch_evaluation_ops(
    *,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_retrieval_config()
    pool = await get_pool(cfg)
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status IN ('pending', 'leased', 'running'))::bigint
                AS queue_depth,
            COUNT(*) FILTER (WHERE status = 'failed')::bigint AS failed_jobs,
            COUNT(*) FILTER (WHERE status = 'dead')::bigint AS dead_jobs,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY EXTRACT(EPOCH FROM (updated_at - created_at))
            ) AS median_job_age_seconds
        FROM retrieval_obs.evaluation_jobs
        """
    )
    lag_row = await pool.fetchrow(
        """
        SELECT
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY EXTRACT(EPOCH FROM (ev.evaluated_at - ev.event_created_at))
            ) AS median_eval_lag_seconds,
            percentile_cont(0.95) WITHIN GROUP (
                ORDER BY EXTRACT(EPOCH FROM (ev.evaluated_at - ev.event_created_at))
            ) AS p95_eval_lag_seconds,
            COUNT(*)::bigint AS completed_evaluations
        FROM retrieval_obs.retrieval_evaluations ev
        WHERE ev.status = 'completed'
          AND ev.evaluated_at >= now() - interval '7 days'
        """
    )
    result = _row_to_json(row) if row else {}
    if lag_row:
        result.update(_row_to_json(lag_row))
    return result


async def fetch_judge_metrics_timeseries(
    filters: EventFilters,
    bucket: BucketGranularity,
    judge: JudgeVersionFilter,
    *,
    config: RetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_retrieval_config()
    trunc = _BUCKET_TRUNC[bucket]
    join, where, params = _eval_where(filters, judge)
    pool = await get_pool(config)
    threshold = cfg.judge_relevance_threshold
    rows = await pool.fetch(
        f"""
        SELECT
            date_trunc('{trunc}', ev.evaluated_at) AS bucket,
            ev.prompt_version,
            ev.judge_model,
            COUNT(*)::bigint AS evaluation_count,
            AVG((ev.aggregated_metrics->>'precision_at_k')::float) AS avg_precision_at_k,
            AVG((ev.aggregated_metrics->>'ndcg_at_k')::float) AS avg_ndcg_at_k,
            AVG(
                CASE
                    WHEN (ev.context_sufficiency->>'sufficient')::boolean THEN 1.0
                    ELSE 0.0
                END
            ) AS context_sufficiency_rate,
            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(ev.per_chunk_scores) elem
                    WHERE (elem->>'relevance')::int >= {threshold}
                )
            )::bigint AS events_with_relevant_chunk
        FROM retrieval_obs.retrieval_evaluations ev
        {join}
        WHERE {where}
        GROUP BY 1, ev.prompt_version, ev.judge_model
        ORDER BY 1, ev.prompt_version, ev.judge_model
        """,
        *params,
    )
    return [_row_to_json(row) for row in rows]


async def fetch_worst_queries(
    filters: EventFilters,
    judge: JudgeVersionFilter,
    *,
    limit: int = 50,
    config: RetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    join, where, params = _eval_where(filters, judge)
    params.append(limit)
    pool = await get_pool(config)
    rows = await pool.fetch(
        f"""
        SELECT
            e.event_id,
            e.created_at,
            COALESCE(
                e.request_payload->>'query',
                e.retrieval_trace->>'query'
            ) AS query_text,
            ev.prompt_version,
            ev.judge_model,
            ev.aggregated_metrics,
            ev.context_sufficiency,
            ev.evaluated_at
        FROM retrieval_obs.retrieval_evaluations ev
        {join}
        WHERE {where}
        ORDER BY (ev.aggregated_metrics->>'ndcg_at_k')::float ASC NULLS LAST,
                 (ev.aggregated_metrics->>'precision_at_k')::float ASC NULLS LAST
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [_row_to_json(row) for row in rows]


async def fetch_zero_result_events(
    filters: EventFilters,
    *,
    limit: int = 100,
    config: RetrievalConfig | None = None,
) -> list[dict[str, Any]]:
    where, params = _event_where(filters)
    params.append(limit)
    pool = await get_pool(config)
    rows = await pool.fetch(
        f"""
        SELECT
            e.event_id,
            e.created_at,
            e.request_source,
            e.index_fingerprint,
            e.embedding_model,
            COALESCE(
                e.request_payload->>'query',
                e.retrieval_trace->>'query'
            ) AS query_text,
            e.duration_ms,
            e.status
        FROM retrieval_obs.retrieval_events e
        WHERE {where}
          AND COALESCE(
                (e.response_payload->>'count')::int,
                jsonb_array_length(COALESCE(e.response_payload->'results', '[]'::jsonb))
          ) = 0
        ORDER BY e.created_at DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [_row_to_json(row) for row in rows]


async def compare_periods(
    period_a: EventFilters,
    period_b: EventFilters,
    judge: JudgeVersionFilter,
    *,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_retrieval_config()

    async def _summary(filters: EventFilters) -> dict[str, Any]:
        where, params = _event_where(filters)
        pool = await get_pool(cfg)
        event_row = await pool.fetchrow(
            f"""
            SELECT
                COUNT(*)::bigint AS event_count,
                COUNT(*) FILTER (
                    WHERE COALESCE(
                        (response_payload->>'count')::int,
                        jsonb_array_length(COALESCE(response_payload->'results', '[]'::jsonb))
                    ) = 0
                )::bigint AS zero_result_count,
                AVG(duration_ms)::float AS avg_duration_ms
            FROM retrieval_obs.retrieval_events e
            WHERE {where}
            """,
            *params,
        )
        latency = await fetch_latency_percentiles(filters, config=cfg)
        coverage = await fetch_coverage(filters, judge, config=cfg)
        return {
            "start": filters.start.isoformat(),
            "end": filters.end.isoformat(),
            "events": _row_to_json(event_row) if event_row else {},
            "latency": latency,
            "coverage": coverage,
        }

    return {
        "period_a": await _summary(period_a),
        "period_b": await _summary(period_b),
        "judge_relevance_threshold": cfg.judge_relevance_threshold,
    }


async def fetch_storage_stats(*, config: RetrievalConfig | None = None) -> dict[str, Any]:
    pool = await get_pool(config)
    tables = await pool.fetch("SELECT * FROM retrieval_obs.table_byte_sizes")
    compression_rows = await pool.fetch(
        "SELECT * FROM retrieval_obs.retrieval_events_compression_stats()"
    )
    return {
        "tables": [_row_to_json(row) for row in tables],
        "compression": [_row_to_json(row) for row in compression_rows],
    }


async def fetch_stats_bundle(
    filters: EventFilters,
    bucket: BucketGranularity,
    judge: JudgeVersionFilter,
    *,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_retrieval_config()
    return {
        "filters": {
            "start": filters.start.isoformat(),
            "end": filters.end.isoformat(),
            "index_fingerprint": filters.index_fingerprint,
            "embedding_model": filters.embedding_model,
            "bucket": bucket,
        },
        "event_timeseries": await fetch_period_event_stats(filters, bucket, config=cfg),
        "latency": await fetch_latency_percentiles(filters, config=cfg),
        "coverage": await fetch_coverage(filters, judge, config=cfg),
        "evaluation_ops": await fetch_evaluation_ops(config=cfg),
        "judge_timeseries": await fetch_judge_metrics_timeseries(
            filters,
            bucket,
            judge,
            config=cfg,
        ),
        "judge_relevance_threshold": cfg.judge_relevance_threshold,
    }


async def list_events_for_bulk_enqueue(
    filters: EventFilters,
    *,
    limit: int,
    cursor_created_at: datetime | None = None,
    cursor_event_id: UUID | str | None = None,
    config: RetrievalConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    where, params = _event_where(filters)
    if cursor_created_at is not None and cursor_event_id is not None:
        params.extend([cursor_created_at, cursor_event_id])
        where = (
            f"{where} AND (e.created_at, e.event_id) > (${len(params) - 1}, ${len(params)})"
        )
    params.append(limit)
    pool = await get_pool(config)
    rows = await pool.fetch(
        f"""
        SELECT e.event_id, e.created_at AS event_created_at
        FROM retrieval_obs.retrieval_events e
        WHERE {where}
        ORDER BY e.created_at, e.event_id
        LIMIT ${len(params)}
        """,
        *params,
    )
    events = [
        {
            "event_id": str(row["event_id"]),
            "event_created_at": row["event_created_at"].isoformat(),
        }
        for row in rows
    ]
    next_cursor: dict[str, Any] | None = None
    if rows and len(rows) == limit:
        last = rows[-1]
        next_cursor = {
            "cursor_created_at": last["event_created_at"].isoformat(),
            "cursor_event_id": str(last["event_id"]),
        }
    return events, next_cursor
