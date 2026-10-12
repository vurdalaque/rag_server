-- Retrieval observability: schema and extensions (idempotent).
-- Python runner: detect capabilities via
--   SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
--   SELECT current_setting('server_version_num')::int;

CREATE SCHEMA IF NOT EXISTS retrieval_obs;

CREATE TABLE IF NOT EXISTS retrieval_obs.schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON SCHEMA retrieval_obs IS
  'Retrieval RAG observability: events (Timescale hypertable), evaluations, async eval jobs.';

-- Optional: record migration runner config (compress_after consumed by 005).
CREATE TABLE IF NOT EXISTS retrieval_obs.migration_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO retrieval_obs.migration_config (key, value)
VALUES ('retrieval_events_compress_after', '7 days')
ON CONFLICT (key) DO NOTHING;

-- TimescaleDB: extension may already exist (managed service) or require superuser.
CREATE EXTENSION IF NOT EXISTS timescaledb;
