-- evaluation_jobs: async worker queue with idempotent enqueue per evaluation_version.

CREATE TABLE IF NOT EXISTS retrieval_obs.evaluation_jobs (
    job_id              BIGSERIAL   PRIMARY KEY,

    event_id            UUID        NOT NULL,
    event_created_at    TIMESTAMPTZ NOT NULL,

    evaluation_version  TEXT        NOT NULL,

    status              TEXT        NOT NULL DEFAULT 'pending',
    attempts            INTEGER     NOT NULL DEFAULT 0,
    lease_expires_at    TIMESTAMPTZ,
    last_error          TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT evaluation_jobs_status_chk
        CHECK (status IN ('pending', 'leased', 'running', 'completed', 'failed', 'dead')),
    CONSTRAINT evaluation_jobs_attempts_nonneg_chk
        CHECK (attempts >= 0)
);

COMMENT ON TABLE retrieval_obs.evaluation_jobs IS
  'Work queue for retrospective evaluations. UNIQUE(event_id, evaluation_version) enables safe bulk re-enqueue.';

-- Idempotency for job creation (same event + evaluation pipeline version).
CREATE UNIQUE INDEX IF NOT EXISTS evaluation_jobs_event_version_uidx
    ON retrieval_obs.evaluation_jobs (event_id, evaluation_version);

CREATE INDEX IF NOT EXISTS evaluation_jobs_pending_lease_idx
    ON retrieval_obs.evaluation_jobs (status, lease_expires_at NULLS FIRST, created_at)
    WHERE status IN ('pending', 'leased', 'running');

CREATE INDEX IF NOT EXISTS evaluation_jobs_event_time_idx
    ON retrieval_obs.evaluation_jobs (event_id, event_created_at);

CREATE INDEX IF NOT EXISTS evaluation_jobs_updated_at_idx
    ON retrieval_obs.evaluation_jobs (updated_at DESC);

-- Bulk (re-)enqueue: insert many rows; skip duplicates on (event_id, evaluation_version).
CREATE OR REPLACE FUNCTION retrieval_obs.enqueue_evaluation_jobs(
    p_evaluation_version TEXT,
    p_events JSONB
)
RETURNS TABLE (
    inserted_count BIGINT,
    skipped_count  BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_inserted BIGINT;
    v_total    BIGINT;
BEGIN
    IF p_events IS NULL OR jsonb_typeof(p_events) <> 'array' THEN
        RAISE EXCEPTION 'p_events must be a JSON array of {event_id, event_created_at} objects';
    END IF;

    v_total := jsonb_array_length(p_events);

    WITH parsed AS (
        SELECT
            (elem->>'event_id')::uuid AS event_id,
            (elem->>'event_created_at')::timestamptz AS event_created_at
        FROM jsonb_array_elements(p_events) AS elem
        WHERE elem ? 'event_id' AND elem ? 'event_created_at'
    ),
    ins AS (
        INSERT INTO retrieval_obs.evaluation_jobs (
            event_id,
            event_created_at,
            evaluation_version,
            status
        )
        SELECT event_id, event_created_at, p_evaluation_version, 'pending'
        FROM parsed
        ON CONFLICT (event_id, evaluation_version) DO NOTHING
        RETURNING 1
    )
    SELECT COUNT(*)::bigint INTO v_inserted FROM ins;

    inserted_count := v_inserted;
    skipped_count := v_total - v_inserted;
    RETURN NEXT;
END;
$$;

COMMENT ON FUNCTION retrieval_obs.enqueue_evaluation_jobs(TEXT, JSONB) IS
  'Bulk enqueue evaluation jobs. Pass JSON array: [{"event_id":"...","event_created_at":"..."}, ...].';

-- Touch updated_at on row change (optional convenience for workers).
CREATE OR REPLACE FUNCTION retrieval_obs.evaluation_jobs_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS evaluation_jobs_updated_at_trg ON retrieval_obs.evaluation_jobs;
CREATE TRIGGER evaluation_jobs_updated_at_trg
    BEFORE UPDATE ON retrieval_obs.evaluation_jobs
    FOR EACH ROW
    EXECUTE PROCEDURE retrieval_obs.evaluation_jobs_set_updated_at();
