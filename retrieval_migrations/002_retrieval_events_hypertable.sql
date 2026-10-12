-- retrieval_events: append-only observability log (Timescale hypertable on created_at).

CREATE TABLE IF NOT EXISTS retrieval_obs.retrieval_events (
    event_id            UUID        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    correlation_id      TEXT,
    request_source      TEXT,
    index_staging_id    TEXT,
    index_fingerprint   TEXT,
    embedding_model     TEXT,

    duration_ms         INTEGER,
    status              TEXT        NOT NULL DEFAULT 'ok',
    error_type          TEXT,

    request_payload     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    response_payload    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    retrieval_trace     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metadata            JSONB       NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT retrieval_events_status_chk
        CHECK (status IN ('ok', 'error', 'partial', 'timeout', 'cancelled')),
    CONSTRAINT retrieval_events_duration_nonneg_chk
        CHECK (duration_ms IS NULL OR duration_ms >= 0)
);

-- Hypertable PK/unique must include partition column (created_at).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class rel ON rel.oid = c.conrelid
        JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
        WHERE nsp.nspname = 'retrieval_obs'
          AND rel.relname = 'retrieval_events'
          AND c.conname = 'retrieval_events_pkey'
    ) THEN
        ALTER TABLE retrieval_obs.retrieval_events
            ADD CONSTRAINT retrieval_events_pkey PRIMARY KEY (event_id, created_at);
    END IF;
END $$;

-- Promote to hypertable when TimescaleDB is present (idempotent).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        IF NOT EXISTS (
            SELECT 1
            FROM timescaledb_information.hypertables h
            WHERE h.hypertable_schema = 'retrieval_obs'
              AND h.hypertable_name = 'retrieval_events'
        ) THEN
            PERFORM create_hypertable(
                'retrieval_obs.retrieval_events',
                'created_at',
                if_not_exists => TRUE,
                migrate_data => FALSE
            );
        END IF;
    ELSE
        RAISE NOTICE 'timescaledb extension not installed; retrieval_events remains a regular partitioned-ready table';
    END IF;
END $$;

COMMENT ON TABLE retrieval_obs.retrieval_events IS
  'One row per retrieval request. Partitioned by created_at when TimescaleDB is enabled.';

-- Time-scoped lookup indexes (leading column + created_at for chunk exclusion).
CREATE INDEX IF NOT EXISTS retrieval_events_correlation_created_idx
    ON retrieval_obs.retrieval_events (correlation_id, created_at DESC)
    WHERE correlation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS retrieval_events_source_created_idx
    ON retrieval_obs.retrieval_events (request_source, created_at DESC)
    WHERE request_source IS NOT NULL;

CREATE INDEX IF NOT EXISTS retrieval_events_staging_created_idx
    ON retrieval_obs.retrieval_events (index_staging_id, created_at DESC)
    WHERE index_staging_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS retrieval_events_fingerprint_created_idx
    ON retrieval_obs.retrieval_events (index_fingerprint, created_at DESC)
    WHERE index_fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS retrieval_events_embedding_model_created_idx
    ON retrieval_obs.retrieval_events (embedding_model, created_at DESC)
    WHERE embedding_model IS NOT NULL;

CREATE INDEX IF NOT EXISTS retrieval_events_status_created_idx
    ON retrieval_obs.retrieval_events (status, created_at DESC);

CREATE INDEX IF NOT EXISTS retrieval_events_created_at_idx
    ON retrieval_obs.retrieval_events (created_at DESC);
