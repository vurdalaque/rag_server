"""Admin HTTP routes for retrieval analytics and export.

Wire into the main app separately, for example::

    from radius_rag.observability.analytics.admin import register_retrieval_admin_routes
    register_retrieval_admin_routes(app, require_admin)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from radius_rag.observability.analytics.analytics import (
    BucketGranularity,
    EventFilters,
    JudgeVersionFilter,
    compare_periods,
    fetch_stats_bundle,
    fetch_storage_stats,
    fetch_worst_queries,
    fetch_zero_result_events,
    list_events_for_bulk_enqueue,
)
from radius_rag.observability.config import load_retrieval_config
from radius_rag.observability.storage.db import enqueue_evaluation_job, enqueue_evaluation_jobs_bulk, fetch_retrieval_event
from radius_rag.observability.analytics.export import ExportCriteria, ExportFormat, export_jsonl


def _parse_dt(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Invalid {name}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_filters(
    start: str,
    end: str,
    index_fingerprint: str | None,
    embedding_model: str | None,
    index_staging_id: str | None = None,
    request_source: str | None = None,
) -> EventFilters:
    return EventFilters(
        start=_parse_dt(start, "start"),
        end=_parse_dt(end, "end"),
        index_fingerprint=index_fingerprint,
        embedding_model=embedding_model,
        index_staging_id=index_staging_id,
        request_source=request_source,
    )


class BulkEvaluateBody(BaseModel):
    start: str
    end: str
    judge_version: str | None = None
    limit: int = Field(default=500, ge=1, le=5000)
    cursor_created_at: str | None = None
    cursor_event_id: str | None = None
    index_fingerprint: str | None = None
    embedding_model: str | None = None


class EvaluateEventBody(BaseModel):
    event_created_at: str


def register_retrieval_admin_routes(
    app: Any,
    require_admin: Callable[..., Any],
) -> None:
    router = APIRouter(prefix="/admin/retrieval", tags=["retrieval-admin"])

    @router.get("/stats")
    async def retrieval_stats(
        request: Request,
        start: str = Query(...),
        end: str = Query(...),
        bucket: BucketGranularity = Query("day"),
        index_fingerprint: str | None = None,
        embedding_model: str | None = None,
        judge_version: str | None = None,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        filters = _event_filters(start, end, index_fingerprint, embedding_model)
        judge = JudgeVersionFilter.parse(judge_version)
        return await fetch_stats_bundle(filters, bucket, judge)

    @router.get("/compare")
    async def retrieval_compare(
        request: Request,
        period_a_start: str = Query(...),
        period_a_end: str = Query(...),
        period_b_start: str = Query(...),
        period_b_end: str = Query(...),
        index_fingerprint: str | None = None,
        embedding_model: str | None = None,
        judge_version: str | None = None,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        period_a = _event_filters(
            period_a_start,
            period_a_end,
            index_fingerprint,
            embedding_model,
        )
        period_b = _event_filters(
            period_b_start,
            period_b_end,
            index_fingerprint,
            embedding_model,
        )
        judge = JudgeVersionFilter.parse(judge_version)
        return await compare_periods(period_a, period_b, judge)

    @router.get("/worst")
    async def retrieval_worst(
        request: Request,
        start: str = Query(...),
        end: str = Query(...),
        limit: int = Query(50, ge=1, le=500),
        judge_version: str | None = None,
        index_fingerprint: str | None = None,
        embedding_model: str | None = None,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        filters = _event_filters(start, end, index_fingerprint, embedding_model)
        judge = JudgeVersionFilter.parse(judge_version)
        rows = await fetch_worst_queries(filters, judge, limit=limit)
        return {"items": rows, "count": len(rows)}

    @router.get("/zero-results")
    async def retrieval_zero_results(
        request: Request,
        start: str = Query(...),
        end: str = Query(...),
        limit: int = Query(100, ge=1, le=1000),
        index_fingerprint: str | None = None,
        embedding_model: str | None = None,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        filters = _event_filters(start, end, index_fingerprint, embedding_model)
        rows = await fetch_zero_result_events(filters, limit=limit)
        return {"items": rows, "count": len(rows)}

    @router.get("/storage")
    async def retrieval_storage(
        request: Request,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        return await fetch_storage_stats()

    # Register /evaluate/bulk before /evaluate/{event_id} so "bulk" is not parsed as UUID.
    @router.post("/evaluate/bulk")
    async def retrieval_evaluate_bulk(
        body: BulkEvaluateBody,
        request: Request,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        cfg = load_retrieval_config()
        filters = _event_filters(
            body.start,
            body.end,
            body.index_fingerprint,
            body.embedding_model,
        )
        cursor_created_at = (
            _parse_dt(body.cursor_created_at, "cursor_created_at")
            if body.cursor_created_at
            else None
        )
        events, next_cursor = await list_events_for_bulk_enqueue(
            filters,
            limit=body.limit,
            cursor_created_at=cursor_created_at,
            cursor_event_id=body.cursor_event_id,
        )
        version = body.judge_version or cfg.evaluation_version
        inserted, skipped = await enqueue_evaluation_jobs_bulk(events, version)
        return {
            "evaluation_version": version,
            "inserted_count": inserted,
            "skipped_count": skipped,
            "next_cursor": next_cursor,
        }

    @router.post("/evaluate/{event_id}")
    async def retrieval_evaluate_one(
        event_id: UUID,
        body: EvaluateEventBody,
        request: Request,
        judge_version: str | None = None,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        cfg = load_retrieval_config()
        created_at = _parse_dt(body.event_created_at, "event_created_at")
        event = await fetch_retrieval_event(event_id, created_at)
        if event is None:
            raise HTTPException(status_code=404, detail="retrieval event not found")

        version = judge_version or cfg.evaluation_version
        inserted = await enqueue_evaluation_job(
            event_id,
            created_at,
            version,
            max_attempts=cfg.max_retries + 1,
            reset_terminal=True,
        )
        return {
            "event_id": str(event_id),
            "event_created_at": created_at.isoformat(),
            "evaluation_version": version,
            "enqueued": inserted,
        }

    @router.get("/export")
    async def retrieval_export(
        request: Request,
        format: ExportFormat = Query("traces", alias="format"),
        start: str = Query(...),
        end: str = Query(...),
        index_fingerprint: str | None = None,
        embedding_model: str | None = None,
        judge_version: str | None = None,
        min_relevance: int = Query(2, ge=0, le=3),
        require_context_sufficient: bool = False,
        _: None = Depends(require_admin),
    ) -> StreamingResponse:
        criteria = ExportCriteria(
            filters=_event_filters(start, end, index_fingerprint, embedding_model),
            judge=JudgeVersionFilter.parse(judge_version),
            min_relevance=min_relevance,
            require_context_sufficient=require_context_sufficient,
        )
        manifest, line_iter = await export_jsonl(format, criteria)

        async def _stream() -> Any:
            yield json.dumps(manifest, ensure_ascii=False) + "\n"
            async for line in line_iter:
                yield line

        filename = f"retrieval_{format}.jsonl"
        return StreamingResponse(
            _stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    app.include_router(router)
