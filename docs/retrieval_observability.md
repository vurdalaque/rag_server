# Retrieval observability and evaluation

## Architecture

```
HTTP/MCP handlers → retrieval_context (correlation, source, payload)
        → rag_service.retrieve()  [ranking unchanged]
        → retrieval_event.maybe_emit_retrieval_event()
        → retrieval_telemetry.emit()  [bounded queue]
        → background writer → TimescaleDB retrieval_events
        → optional evaluation_jobs enqueue

python -m retrieval_evaluator  → evaluation worker → LLM judge → retrieval_evaluations
```

Admin JSON API (`RAG_ADMIN_TOKEN`): `/admin/retrieval/*` — stats, compare, export, bulk/single re-eval enqueue.

## Enable telemetry

```env
RETRIEVAL_TELEMETRY_ENABLED=true
RETRIEVAL_DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Migrations run on server startup when telemetry is enabled and DSN is set.

## Evaluator process

```bash
cd rag_server
export RETRIEVAL_DATABASE_URL=...
export RETRIEVAL_EVALUATION_AUTO_ENQUEUE=true   # optional: enqueue on each event
python -m retrieval_evaluator
```

Concurrency: `RETRIEVAL_EVALUATOR_CONCURRENCY` (default 1).

## Telemetry loss scenarios

Events may be **lost** (never persisted) when:

1. `RETRIEVAL_TELEMETRY_ENABLED=false` or `RETRIEVAL_DATABASE_URL` unset — emit is a no-op.
2. Telemetry writer not started (misconfiguration) — queue is `None`, emit returns false.
3. **Queue full** (`RETRIEVAL_TELEMETRY_QUEUE_SIZE`, default 1000) — `emit` drops and increments `retrieval_telemetry_queue_drops_total`.
4. **Batch insert failure** (DB down, constraint error, network) — batch is logged and discarded; retrieval is unaffected.
4b. **Enqueue failure after successful insert** — events remain in DB; jobs can be re-enqueued via admin bulk API (`retrieval_telemetry_enqueue_errors_total`).
5. **Process crash / kill -9** before graceful shutdown drain — queued events not flushed.
6. **Shutdown timeout** (30s) waiting for writer — remaining queue may be cancelled.

Events are **not** lost when DB is temporarily slow if the queue has capacity; they are lost once the queue fills.

## Security note

`GET /debug/search` remains unauthenticated; it is logged with `request_source=debug`. Consider protecting it separately.
