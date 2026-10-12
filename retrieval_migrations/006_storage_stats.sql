-- Optional storage / compression observability helpers.

CREATE OR REPLACE VIEW retrieval_obs.table_byte_sizes AS
SELECT
    n.nspname AS schema_name,
    c.relname AS relation_name,
    pg_total_relation_size(c.oid) AS total_bytes,
    pg_relation_size(c.oid) AS heap_bytes,
    pg_indexes_size(c.oid) AS index_bytes,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_pretty
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'retrieval_obs'
  AND c.relkind IN ('r', 'p')
  AND c.relname IN ('retrieval_events', 'retrieval_evaluations', 'evaluation_jobs')
ORDER BY pg_total_relation_size(c.oid) DESC;

COMMENT ON VIEW retrieval_obs.table_byte_sizes IS
  'Approximate on-disk sizes for retrieval observability tables (includes indexes and TOAST).';

CREATE OR REPLACE FUNCTION retrieval_obs.retrieval_events_compression_stats()
RETURNS TABLE (
    hypertable_schema TEXT,
    hypertable_name   TEXT,
    total_chunks      BIGINT,
    compressed_chunks BIGINT,
    uncompressed_chunks BIGINT,
    compression_ratio NUMERIC
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        h.hypertable_schema::text,
        h.hypertable_name::text,
        COUNT(*)::bigint AS total_chunks,
        COUNT(*) FILTER (WHERE ch.compression_status = 'Compressed')::bigint AS compressed_chunks,
        COUNT(*) FILTER (WHERE ch.compression_status IS DISTINCT FROM 'Compressed')::bigint AS uncompressed_chunks,
        CASE
            WHEN COUNT(*) FILTER (WHERE ch.compression_status = 'Compressed') = 0 THEN NULL
            ELSE ROUND(
                COUNT(*)::numeric
                / NULLIF(COUNT(*) FILTER (WHERE ch.compression_status = 'Compressed'), 0)::numeric,
                4
            )
        END AS compression_ratio
    FROM timescaledb_information.hypertables h
    LEFT JOIN timescaledb_information.chunks ch
        ON ch.hypertable_schema = h.hypertable_schema
       AND ch.hypertable_name = h.hypertable_name
    WHERE h.hypertable_schema = 'retrieval_obs'
      AND h.hypertable_name = 'retrieval_events'
    GROUP BY h.hypertable_schema, h.hypertable_name;
END;
$$;

COMMENT ON FUNCTION retrieval_obs.retrieval_events_compression_stats() IS
  'Chunk compression breakdown for retrieval_events; empty when TimescaleDB is absent.';
