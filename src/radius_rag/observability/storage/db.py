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


def _validate_database_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        raise RuntimeError("RETRIEVAL_DATABASE_URL is not configured")
    lowered = normalized.lower()
    if lowered.startswith("postgresql//") or lowered.startswith("postgres//"):
        raise RuntimeError(
            "RETRIEVAL_DATABASE_URL is invalid: use postgresql://user:pass@host:port/dbname "
            "(note the colon and two slashes after postgresql)"
        )
    if not (lowered.startswith("postgresql://") or lowered.startswith("postgres://")):
        raise RuntimeError(
            "RETRIEVAL_DATABASE_URL must start with postgresql:// or postgres://"
        )
    return normalized


async def get_pool(config: RetrievalConfig | None = None) -> Any:
    global _pool
    cfg = config or load_retrieval_config()
    dsn = _validate_database_url(cfg.database_url)
    if _pool is None:
        asyncpg = load_asyncpg()
        _pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            command_timeout=120,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is None:
        return
    pool = _pool
    _pool = None
    try:
        await pool.close()
    except Exception:
        logger.debug("asyncpg pool close failed", exc_info=True)


def discard_pool() -> None:
    """Release pool reference without closing (safe when the event loop is gone)."""
    global _pool
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
    reset_terminal: bool = False,
) -> bool:
    pool = await get_pool()
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    max_att = max_attempts if max_attempts is not None else 3

    async with pool.acquire() as connection:
        if reset_terminal:
            reset = await connection.execute(
                """
                UPDATE retrieval_obs.evaluation_jobs
                SET status = 'pending',
                    attempts = 0,
                    lease_expires_at = NULL,
                    lease_token = NULL,
                    next_attempt_at = NULL,
                    last_error = NULL,
                    event_created_at = $3,
                    max_attempts = $4,
                    updated_at = now()
                WHERE event_id = $1
                  AND evaluation_version = $2
                  AND status IN ('failed', 'dead')
                """,
                event_id,
                evaluation_version,
                event_created_at,
                max_att,
            )
            if reset.endswith("1"):
                return True

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


async def reconcile_missing_evaluation_jobs(
    evaluation_version: str,
    *,
    limit: int = 100,
    max_attempts: int = 3,
    event_id: UUID | str | None = None,
) -> int:
    """Create pending jobs for stored events that have no job for this evaluator version."""
    if limit <= 0:
        return 0
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    pool = await get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            INSERT INTO retrieval_obs.evaluation_jobs (
                event_id, event_created_at, evaluation_version, status, max_attempts
            )
            SELECT e.event_id, e.created_at, $1, 'pending', $3
            FROM retrieval_obs.retrieval_events e
            WHERE NOT EXISTS (
                SELECT 1
                FROM retrieval_obs.evaluation_jobs j
                WHERE j.event_id = e.event_id
                  AND j.evaluation_version = $1
            )
              AND ($4::uuid IS NULL OR e.event_id = $4)
            ORDER BY e.created_at DESC
            LIMIT $2
            ON CONFLICT (event_id, evaluation_version) DO NOTHING
            RETURNING job_id
            """,
            evaluation_version,
            limit,
            max_attempts,
            event_id,
        )
    return len(rows)


async def requeue_failed_evaluation_jobs(
    evaluation_version: str,
    *,
    skip_data_errors: bool = True,
    include_dead: bool = True,
) -> int:
    """Reset failed/dead jobs to pending for another worker attempt."""
    statuses = ["failed", "dead"] if include_dead else ["failed"]
    pool = await get_pool()
    skip_clause = ""
    if skip_data_errors:
        skip_clause = """
          AND last_error NOT ILIKE '%empty_final_top_k%'
          AND last_error NOT ILIKE '%missing_chunk_text%'
        """
    result = await pool.execute(
        f"""
        UPDATE retrieval_obs.evaluation_jobs
        SET status = 'pending',
            attempts = 0,
            lease_expires_at = NULL,
            lease_token = NULL,
            next_attempt_at = NULL,
            last_error = NULL,
            updated_at = now()
        WHERE evaluation_version = $1
          AND status = ANY($2::text[])
          {skip_clause}
        """,
        evaluation_version,
        statuses,
    )
    # asyncpg returns e.g. "UPDATE 12"
    try:
        return int(result.split()[-1])
    except (AttributeError, IndexError, ValueError):
        return 0


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


# Jobs on their final attempt keep running until lease expires or worker finishes.
_SQL_TERMINALIZE_EXHAUSTED_JOBS = """
UPDATE retrieval_obs.evaluation_jobs
SET status = 'dead',
    lease_expires_at = NULL,
    lease_token = NULL,
    last_error = COALESCE(last_error, 'max attempts exceeded'),
    updated_at = now()
WHERE evaluation_version = $1
  AND status IN ('pending', 'leased', 'running')
  AND attempts >= max_attempts
  AND NOT (
    status IN ('leased', 'running')
    AND lease_expires_at IS NOT NULL
    AND lease_expires_at > now()
  )
"""


async def mark_exhausted_evaluation_jobs(evaluation_version: str) -> int:
    """Terminalize jobs that exceeded max_attempts without an active lease."""
    pool = await get_pool()
    async with pool.acquire() as connection:
        result = await connection.execute(
            _SQL_TERMINALIZE_EXHAUSTED_JOBS,
            evaluation_version,
        )
    # asyncpg returns "UPDATE N"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


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
            await connection.execute(
                _SQL_TERMINALIZE_EXHAUSTED_JOBS,
                evaluation_version,
            )
            rows = await connection.fetch(
                """
                SELECT job_id, event_id, event_created_at, evaluation_version, attempts, max_attempts
                FROM retrieval_obs.evaluation_jobs
                WHERE evaluation_version = $1
                  AND status IN ('pending', 'leased', 'running')
                  AND attempts < max_attempts
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


class EvaluationCommitLeaseError(RuntimeError):
    """Job lease invalid or lost; transaction rolled back (no evaluation persisted)."""


async def commit_evaluation_and_complete_job(
    job_id: int,
    lease_token: UUID,
    row: dict[str, Any],
) -> bool:
    """Atomically verify lease, insert evaluation (idempotent), and complete the job."""
    pool = await get_pool()
    evaluation_id = row.get("evaluation_id")
    if evaluation_id is not None and isinstance(evaluation_id, str):
        evaluation_id = UUID(evaluation_id)

    event_id = row["event_id"]
    if isinstance(event_id, str):
        event_id = UUID(event_id)

    prompt_version = row["prompt_version"]
    judge_model = row["judge_model"]
    evaluation_version = row.get("evaluation_version")
    if not evaluation_version:
        from radius_rag.observability.evaluation.identity import build_evaluation_version

        evaluation_version = build_evaluation_version(
            prompt_version,
            judge_model,
            row.get("methodology_fingerprint"),
        )

    async with pool.acquire() as connection:
        async with connection.transaction():
            job = await connection.fetchrow(
                """
                SELECT job_id, event_id, event_created_at, evaluation_version,
                       status, lease_expires_at
                FROM retrieval_obs.evaluation_jobs
                WHERE job_id = $1 AND lease_token = $2
                FOR UPDATE
                """,
                job_id,
                lease_token,
            )
            if job is None:
                return False

            if job["status"] not in ("running", "leased"):
                return False

            lease_expires = job["lease_expires_at"]
            if lease_expires is not None and lease_expires <= _utcnow():
                return False

            if job["event_id"] != event_id:
                raise EvaluationCommitLeaseError("event_id does not match job")

            if job["evaluation_version"] != evaluation_version:
                raise EvaluationCommitLeaseError("evaluation_version does not match job")

            existing = await connection.fetchval(
                """
                SELECT evaluation_id
                FROM retrieval_obs.retrieval_evaluations
                WHERE event_id = $1
                  AND event_created_at = $2
                  AND evaluation_version = $3
                  AND status = 'completed'
                LIMIT 1
                """,
                event_id,
                row["event_created_at"],
                evaluation_version,
            )
            if existing is None:
                eval_id = evaluation_id or await connection.fetchval("SELECT gen_random_uuid()")
                await connection.execute(
                    """
                    INSERT INTO retrieval_obs.retrieval_evaluations (
                        evaluation_id, event_id, event_created_at,
                        evaluation_version, judge_model, prompt_version,
                        per_chunk_scores, context_sufficiency, query_classification,
                        aggregated_metrics, explanation, status, diagnostics, truncation_info
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb,
                        $11, $12, $13::jsonb, $14::jsonb
                    )
                    """,
                    eval_id,
                    event_id,
                    row["event_created_at"],
                    evaluation_version,
                    judge_model,
                    prompt_version,
                    json.dumps(row.get("per_chunk_scores") or []),
                    json.dumps(row.get("context_sufficiency") or {}),
                    json.dumps(row.get("query_classification") or {}),
                    json.dumps(row.get("aggregated_metrics") or {}),
                    row.get("explanation"),
                    row.get("status", "completed"),
                    json.dumps(row.get("diagnostics") or {}),
                    json.dumps(row.get("truncation_info") or {}),
                )

            result = await connection.execute(
                """
                UPDATE retrieval_obs.evaluation_jobs
                SET status = 'completed',
                    lease_expires_at = NULL,
                    lease_token = NULL,
                    last_error = NULL,
                    next_attempt_at = NULL,
                    updated_at = now()
                WHERE job_id = $1
                  AND lease_token = $2
                  AND status IN ('running', 'leased')
                  AND (lease_expires_at IS NULL OR lease_expires_at > now())
                """,
                job_id,
                lease_token,
            )
            if not result.endswith("1"):
                raise EvaluationCommitLeaseError("lease lost before job completion")
    return True


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

    evaluation_version = row.get("evaluation_version")
    if not evaluation_version:
        from radius_rag.observability.evaluation.identity import build_evaluation_version

        evaluation_version = build_evaluation_version(
            row["prompt_version"],
            row["judge_model"],
        )

    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO retrieval_obs.retrieval_evaluations (
                evaluation_id, event_id, event_created_at,
                evaluation_version, judge_model, prompt_version,
                per_chunk_scores, context_sufficiency, query_classification,
                aggregated_metrics, explanation, status, diagnostics, truncation_info
            )
            VALUES (
                $1, $2, $3, $4, $5, $6,
                $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb,
                $11, $12, $13::jsonb, $14::jsonb
            )
            """,
            evaluation_id,
            event_id,
            row["event_created_at"],
            evaluation_version,
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
