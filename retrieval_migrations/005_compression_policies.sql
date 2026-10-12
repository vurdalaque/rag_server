-- Compression for retrieval_events (TimescaleDB 2.x+ only).
-- compress_after interval: retrieval_obs.migration_config key retrieval_events_compress_after
-- Alternate template without auto-apply: 005_compression_policies.sql.template

DO $$
DECLARE
    v_ext_version   TEXT;
    v_major         INT;
    v_compress_after INTERVAL;
    v_hypertable    BOOLEAN;
    v_job_exists    BOOLEAN;
BEGIN
    SELECT extversion INTO v_ext_version FROM pg_extension WHERE extname = 'timescaledb';
    IF v_ext_version IS NULL THEN
        RAISE NOTICE '005_compression_policies: skipped (timescaledb extension not installed)';
        RETURN;
    END IF;

    v_major := split_part(v_ext_version, '.', 1)::INT;
    IF v_major < 2 THEN
        RAISE NOTICE '005_compression_policies: skipped (timescaledb % < 2.0; add_compression_policy unavailable)', v_ext_version;
        RETURN;
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables h
        WHERE h.hypertable_schema = 'retrieval_obs'
          AND h.hypertable_name = 'retrieval_events'
    ) INTO v_hypertable;

    IF NOT v_hypertable THEN
        RAISE NOTICE '005_compression_policies: skipped (retrieval_events is not a hypertable)';
        RETURN;
    END IF;

    SELECT COALESCE(
        (SELECT value::interval FROM retrieval_obs.migration_config WHERE key = 'retrieval_events_compress_after'),
        INTERVAL '7 days'
    ) INTO v_compress_after;

    -- Enable columnar compression (idempotent settings).
    BEGIN
        ALTER TABLE retrieval_obs.retrieval_events SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'request_source',
            timescaledb.compress_orderby = 'created_at DESC'
        );
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE '005_compression_policies: compress settings already applied or unsupported: %', SQLERRM;
    END;

    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.jobs j
        WHERE j.hypertable_schema = 'retrieval_obs'
          AND j.hypertable_name = 'retrieval_events'
          AND j.proc_name = 'policy_compression'
    ) INTO v_job_exists;

    IF v_job_exists THEN
        RAISE NOTICE '005_compression_policies: compression policy job already exists';
        RETURN;
    END IF;

    PERFORM add_compression_policy(
        'retrieval_obs.retrieval_events',
        v_compress_after,
        if_not_exists => TRUE
    );

    RAISE NOTICE '005_compression_policies: compression policy added (compress_after=%)', v_compress_after;
END $$;
