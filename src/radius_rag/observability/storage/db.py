"""Async PostgreSQL access for retrieval observability."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from radius_rag.observability.config import RetrievalConfig, load_retrieval_config
from radius_rag.observability.storage._asyncpg import load_asyncpg

logger = logging.getLogger(__name__)

_pool: Any = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def get_pool(config: RetrievalConfig | None = None) -> Any:
    global _pool
    cfg = config or load_retrieval_config()
    if not cfg.database_url:
        raise RuntimeError("RETRIEVAL_DATABASE_URL is not configured")
    if _pool is None:
        asyncpg = load_asyncpg()
        _pool = await asyncpg.create_pool(
            cfg.database_url,
            min_size=1,
            max_size=5,
            command_timeout=120,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _migration_files(migrations_dir: Path) -> list[Path]:
    files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))
    return [path for path in files if not path.name.endswith(".template")]


async def _fetch_applied_versions(connection: Any) -> set[str]:
    asyncpg = load_asyncpg()
    try:
        rows = await connection.fetch(
            "SELECT version FROM retrieval_obs.schema_migrations ORDER BY version"
        )
        return {row["version"] for row in rows}
    except asyncpg.UndefinedTableError:
        return set()
    except asyncpg.InvalidSchemaNameError:
        return set()


async def run_migrations(config: RetrievalConfig | None = None) -> list[str]:
    cfg = config or load_retrieval_config()
    pool = await get_pool(cfg)
    applied: list[str] = []

    async with pool.acquire() as connection:
        await connection.execute("CREATE SCHEMA IF NOT EXISTS retrieval_obs")
        already = await _fetch_applied_versions(connection)

        for path in _migration_files(cfg.migrations_dir):
            version = path.stem
            if version in already:
                continue
            sql = path.read_text(encoding="utf-8")
            logger.info("Applying retrieval migration %s", version)
            try:
                async with connection.transaction():
                    await connection.execute(sql)
                    await connection.execute(
                        """
                        INSERT INTO retrieval_obs.schema_migrations (version)
                        VALUES ($1)
                        ON CONFLICT (version) DO NOTHING
                        """,
                        version,
                    )
            except Exception:
                logger.exception("Migration %s failed", version)
                raise
            applied.append(version)
            logger.info("Applied retrieval migration %s", version)

    return applied


async def insert_events(events: list[dict[str, Any]]) -> int:
    """Insert events idempotently; returns number of newly inserted rows."""
    if not events:
        return 0

    pool = await get_pool()
    inserted = 0
    statement = """
        INSERT INTO retrieval_obs.retrieval_events (
            event_id, created_at, correlation_id, request_source,
            index_staging_id, index_fingerprint, embedding_model,
            duration_ms, status, error_type,
            request_payload, response_payload, retrieval_trace, metadata
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb
        )
        ON CONFLICT (event_id, created_at) DO NOTHING
        RETURNING event_id
    """

    async with pool.acquire() as connection:
        for event in events:
            created_at = event.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            elif created_at is None:
                created_at = _utcnow()

            event_id = event.get("event_id")
            if isinstance(event_id, str):
                event_id = UUID(event_id)

            row = await connection.fetchrow(
                statement,
                event_id,
                created_at,
                event.get("correlation_id"),
                event.get("request_source"),
                event.get("index_staging_id"),
                event.get("index_fingerprint"),
                event.get("embedding_model"),
                event.get("duration_ms"),
                event.get("status", "ok"),
                event.get("error_type"),
                json.dumps(event.get("request_payload") or {}),
                json.dumps(event.get("response_payload") or {}),
                json.dumps(event.get("retrieval_trace") or {}),
                json.dumps(event.get("metadata") or {}),
            )
            if row is not None:
                inserted += 1
    return inserted


async def enqueue_evaluation_job(
    event_id: UUID | str,
    event_created_at: datetime,
    evaluation_version: str,
    *,
    max_attempts: int | None = None,
) -> bool:
    pool = await get_pool()
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    max_att = max_attempts if max_attempts is not None else 3

    async with pool.acquire() as connection:
        status = await connection.fetchval(
            """
            INSERT INTO retrieval_obs.evaluation_jobs (
                event_id, event_created_at, evaluation_version, status, max_attempts
            )
            VALUES ($1, $2, $3, 'pending', $4)
            ON CONFLICT (event_id, evaluation_version) DO NOTHING
            RETURNING job_id
            """,
            event_id,
            event_created_at,
            evaluation_version,
            max_att,
        )
    return status is not None


async def enqueue_evaluation_jobs_bulk(
    events: list[dict[str, Any]],
    evaluation_version: str,
) -> tuple[int, int]:
    if not events:
        return 0, 0
    pool = await get_pool()
    payload = json.dumps(events)
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT * FROM retrieval_obs.enqueue_evaluation_jobs($1, $2::jsonb)",
            evaluation_version,
            payload,
        )
    if row is None:
        return 0, len(events)
    return int(row["inserted_count"]), int(row["skipped_count"])


async def claim_evaluation_jobs(
    evaluation_version: str,
    limit: int,
    lease_ttl_seconds: int,
    worker_id: str,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    lease_until = _utcnow() + timedelta(seconds=lease_ttl_seconds)

    async with pool.acquire() as connection:
        async with connection.transaction():
            rows = await connection.fetch(
                """
                SELECT job_id, event_id, event_created_at, evaluation_version, attempts, max_attempts
                FROM retrieval_obs.evaluation_jobs
                WHERE evaluation_version = $1
                  AND status IN ('pending', 'leased', 'running')
                  AND (next_attempt_at IS NULL OR next_attempt_at <= now())
                  AND (lease_expires_at IS NULL OR lease_expires_at < now())
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT $2
                """,
                evaluation_version,
                limit,
            )
            if not rows:
                return []

            claimed: list[dict[str, Any]] = []
            for row in rows:
                token = await connection.fetchval(
                    """
                    UPDATE retrieval_obs.evaluation_jobs
                    SET status = 'leased',
                        lease_expires_at = $2,
                        lease_token = gen_random_uuid(),
                        attempts = attempts + 1,
                        last_error = NULL,
                        updated_at = now()
                    WHERE job_id = $1
                    RETURNING lease_token
                    """,
                    row["job_id"],
                    lease_until,
                )
                if token is None:
                    continue
                claimed.append(
                    {
                        "job_id": row["job_id"],
                        "event_id": row["event_id"],
                        "event_created_at": row["event_created_at"],
                        "evaluation_version": row["evaluation_version"],
                        "attempts": int(row["attempts"]) + 1,
                        "max_attempts": int(row["max_attempts"]),
                        "lease_token": token,
                        "worker_id": worker_id,
                    }
                )
    return claimed


async def mark_evaluation_job_running(
    job_id: int,
    lease_token: UUID,
    lease_ttl_seconds: int,
) -> bool:
    pool = await get_pool()
    lease_until = _utcnow() + timedelta(seconds=lease_ttl_seconds)
    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET status = 'running',
                lease_expires_at = $3,
                updated_at = now()
            WHERE job_id = $1 AND lease_token = $2
            """,
            job_id,
            lease_token,
            lease_until,
        )
    return result.endswith("1")


async def extend_evaluation_job_lease(
    job_id: int,
    lease_token: UUID,
    lease_ttl_seconds: int,
) -> bool:
    pool = await get_pool()
    lease_until = _utcnow() + timedelta(seconds=lease_ttl_seconds)
    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET lease_expires_at = $3,
                updated_at = now()
            WHERE job_id = $1 AND lease_token = $2 AND status = 'running'
            """,
            job_id,
            lease_token,
            lease_until,
        )
    return result.endswith("1")


async def finish_evaluation_job(
    job_id: int,
    lease_token: UUID,
    *,
    status: str,
    error: str | None = None,
    next_attempt_at: datetime | None = None,
) -> bool:
    pool = await get_pool()
    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE retrieval_obs.evaluation_jobs
            SET status = $3,
                lease_expires_at = NULL,
                lease_token = NULL,
                last_error = $4,
                next_attempt_at = $5,
                updated_at = now()
            WHERE job_id = $1 AND lease_token = $2
            """,
            job_id,
            lease_token,
            status,
            error,
            next_attempt_at,
        )
    return result.endswith("1")


async def fetch_retrieval_event(
    event_id: UUID | str,
    event_created_at: datetime,
) -> Any | None:
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    pool = await get_pool()
    async with pool.acquire() as connection:
        return await connection.fetchrow(
            """
            SELECT *
            FROM retrieval_obs.retrieval_events
            WHERE event_id = $1 AND created_at = $2
            LIMIT 1
            """,
            event_id,
            event_created_at,
        )


async def insert_evaluation(row: dict[str, Any]) -> UUID:
    pool = await get_pool()
    evaluation_id = row.get("evaluation_id")
    if evaluation_id is None:
        evaluation_id = await pool.fetchval("SELECT gen_random_uuid()")
    elif isinstance(evaluation_id, str):
        evaluation_id = UUID(evaluation_id)

    event_id = row["event_id"]
    if isinstance(event_id, str):
        event_id = UUID(event_id)

    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO retrieval_obs.retrieval_evaluations (
                evaluation_id, event_id, event_created_at,
                judge_model, prompt_version,
                per_chunk_scores, context_sufficiency, query_classification,
                aggregated_metrics, explanation, status, diagnostics, truncation_info
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb,
                $10, $11, $12::jsonb, $13::jsonb
            )
            """,
            evaluation_id,
            event_id,
            row["event_created_at"],
            row["judge_model"],
            row["prompt_version"],
            json.dumps(row.get("per_chunk_scores") or []),
            json.dumps(row.get("context_sufficiency") or {}),
            json.dumps(row.get("query_classification") or {}),
            json.dumps(row.get("aggregated_metrics") or {}),
            row.get("explanation"),
            row.get("status", "completed"),
            json.dumps(row.get("diagnostics") or {}),
            json.dumps(row.get("truncation_info") or {}),
        )
    return evaluation_id
