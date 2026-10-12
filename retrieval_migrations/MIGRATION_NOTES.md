# Retrieval observability SQL migrations

Migrations live in this directory as ordered `NNN_*.sql` files. Apply in lexical order with a small Python runner (not included here) that records applied revisions in a table such as `retrieval_obs.schema_migrations(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ)`.

## Version detection (required)

Before applying Timescale-specific steps, query:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';
SELECT current_setting('server_version_num')::int AS server_version_num;
```

Runner behavior:

| Condition | Action |
|-----------|--------|
| No `timescaledb` extension | Run `001`–`004`, `006`; skip `005` (file no-ops with NOTICE) or skip entirely. `002` leaves `retrieval_events` as a normal table with time-oriented indexes. |
| TimescaleDB major &lt; 2 | Skip compression (`005` self-skips). Hypertable creation in `002` still runs if extension exists. |
| TimescaleDB ≥ 2 and hypertable present | Run `005` or set `retrieval_events_compress_after` then use `005_compression_policies.sql.template`. |

`CREATE EXTENSION timescaledb` in `001` may require superuser; on managed Timescale Cloud the extension is usually preinstalled—treat extension errors as non-fatal if the extension already exists.

## Timescale / hypertable constraints

- **Partition key**: `retrieval_events.created_at` (hypertable dimension).
- **Primary / unique keys** on hypertables must include the partition column. `retrieval_events` uses `PRIMARY KEY (event_id, created_at)`.
- **No FK** from `retrieval_evaluations` or `evaluation_jobs` to `retrieval_events`: foreign keys across hypertables are awkward and often unsupported; linkage is `(event_id, event_created_at)` enforced in application code.
- **Indexes**: leading filter columns plus `created_at DESC` (or `evaluated_at`) for chunk exclusion and time-range scans.
- **Judge fields** live only on `retrieval_evaluations` (`judge_model`, scores, etc.), not on `retrieval_events`.

## Idempotency

- Tables / indexes: `IF NOT EXISTS` where supported.
- Hypertable: `create_hypertable(..., if_not_exists => TRUE)` guarded by `timescaledb_information.hypertables`.
- **Jobs**: `UNIQUE (event_id, evaluation_version)` on `evaluation_jobs`; bulk enqueue via `retrieval_obs.enqueue_evaluation_jobs(version, jsonb_array)`.
- **Evaluations**: multiple rows per event allowed; dedupe at job layer or in app if needed.

## Compression

- Default `compress_after`: `7 days` in `retrieval_obs.migration_config` (`retrieval_events_compress_after`).
- `005_compression_policies.sql` calls `add_compression_policy` only when TimescaleDB ≥ 2 and `retrieval_events` is a hypertable; uses `if_not_exists => TRUE` when supported.
- To change retention/compression delay before applying policy, update `migration_config` or run the `.template` file manually.

## Suggested Python runner outline

1. Connect with autocommit off; optional `pg_advisory_lock` for single-flyer migrations.
2. Ensure `retrieval_obs.schema_migrations` exists.
3. For each unapplied `NNN_*.sql` (exclude `*.template`):
   - Execute file in a transaction.
   - Insert version row on success.
4. Log `RAISE NOTICE` from `005` when compression is skipped.

## Storage helpers

- `retrieval_obs.table_byte_sizes` — relation sizes for core tables.
- `retrieval_obs.retrieval_events_compression_stats()` — chunk compression summary when TimescaleDB is present.
