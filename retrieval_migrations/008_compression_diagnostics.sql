-- Record compression policy outcome for operators (does not alter hypertable data).

INSERT INTO retrieval_obs.migration_config (key, value, updated_at)
VALUES ('compression_policy_last_check', now()::text, now())
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now();

DO $$
DECLARE
    v_ext_version TEXT;
    v_status TEXT := 'unknown';
    v_job_exists BOOLEAN;
BEGIN
    SELECT extversion INTO v_ext_version FROM pg_extension WHERE extname = 'timescaledb';
    IF v_ext_version IS NULL THEN
        v_status := 'skipped_no_timescaledb';
    ELSIF split_part(v_ext_version, '.', 1)::INT < 2 THEN
        v_status := 'skipped_timescaledb_lt_2:' || v_ext_version;
    ELSE
        SELECT EXISTS (
            SELECT 1 FROM timescaledb_information.jobs j
            WHERE j.hypertable_schema = 'retrieval_obs'
              AND j.hypertable_name = 'retrieval_events'
              AND j.proc_name = 'policy_compression'
        ) INTO v_job_exists;
        IF v_job_exists THEN
            v_status := 'compression_policy_active:' || v_ext_version;
        ELSE
            v_status := 'compression_policy_missing:' || v_ext_version;
        END IF;
    END IF;

    INSERT INTO retrieval_obs.migration_config (key, value, updated_at)
    VALUES ('compression_policy_status', v_status, now())
    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now();
END $$;
