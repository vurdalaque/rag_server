# Module relocation map

## Observability (done)

| Legacy shim (repo root) | Implementation |
|-------------------------|----------------|
| `retrieval_config.py` | `src/radius_rag/observability/config.py` |
| `retrieval_context.py` | `observability/context.py` |
| `retrieval_candidate.py` | `observability/candidate.py` |
| `retrieval_sanitize.py` | `observability/sanitize.py` |
| `retrieval_snapshot.py` | `observability/snapshot.py` |
| `retrieval_trace.py` | `observability/trace.py` |
| `retrieval_index_info.py` | `observability/index_info.py` |
| `retrieval_event.py` | `observability/event.py` |
| `retrieval_telemetry.py` | `observability/telemetry/writer.py` |
| `retrieval_telemetry_metrics.py` | `observability/telemetry/metrics.py` |
| `retrieval_db.py` | `observability/storage/db.py` |
| `retrieval_judge.py` | `observability/evaluation/judge.py` |
| `retrieval_eval_worker.py` | `observability/evaluation/worker.py` |
| `retrieval_evaluator.py` | `observability/evaluation/evaluator.py` |
| `retrieval_quality_metrics.py` | `observability/evaluation/quality_metrics.py` |
| `retrieval_analytics.py` | `observability/analytics/analytics.py` |
| `retrieval_export.py` | `observability/analytics/export.py` |
| `retrieval_admin.py` | `observability/analytics/admin.py` |

Migrations stay at `retrieval_migrations/` (override with `RETRIEVAL_MIGRATIONS_DIR`).

Entrypoints:

- `python -m retrieval_evaluator` (shim)
- `radius-rag-evaluator` → `radius_rag.observability.evaluation.entrypoint`

## PYTHONPATH

- Tests: `conftest.py` adds `src/`
- Production: `start.sh` sets `PYTHONPATH=$ROOT/src`

## RAG (done)

| Legacy shim | Implementation |
|-------------|----------------|
| `rag_metrics.py` | `src/radius_rag/rag/metrics.py` |
| `rag_bundle.py` | `radius_rag/rag/bundle.py` |
| `rag_service.py` | `radius_rag/rag/service.py` |
| `rag_search.py` | `radius_rag/rag/search.py` |
| `rag_proxy.py` | `radius_rag/rag/proxy.py` |

Regenerate: `python tools/migrate_rag_package.py` + `python tools/write_rag_shims.py`.

## LLM (done)

| Legacy shim | Implementation |
|-------------|----------------|
| `llm_params.py` | `src/radius_rag/llm/params.py` |
| `llm_client.py` | `src/radius_rag/llm/client.py` |

Regenerate: `python tools/migrate_llm_package.py` + `python tools/write_llm_shims.py`.

## API (done)

| Legacy shim | Implementation |
|-------------|----------------|
| `rag_server.py` | `src/radius_rag/api/server.py` (`uvicorn rag_server:app` unchanged) |

Regenerate: `python tools/migrate_api_package.py`.

## Imaging (done)

Root `image_*` and `comfy_*` → `src/radius_rag/imaging/<same module name>.py`.

## MCP image tools (done)

`image_mcp_*` → `src/radius_rag/mcp/`. MCP test helpers live in `tests/helpers/mcp_test_helpers.py` only.

Regenerate: `python tools/migrate_imaging_mcp_package.py` + `python tools/write_imaging_mcp_shims.py`.

## Tests

- `tests/unit/` — fast mocks, retrieval/imaging/rag units
- `tests/integration/` — `TestClient`, optional `RETRIEVAL_DATABASE_URL`
- `tests/e2e/` — live uvicorn / ComfyUI (`acceptance` marker)

`pytest` collects only `tests/` (`pyproject.toml`). Helpers: `tests/helpers/`.

Regenerate shims: `python tools/migrate_*_package.py` + `python tools/write_*_shims.py`.

## Root Python files

| File | Purpose |
|------|---------|
| `rag_server.py` | Uvicorn entry (`rag_server:app`). |
| `retrieval_evaluator.py` | `python -m retrieval_evaluator`. |
| `conftest.py` | Pytest: `PYTHONPATH=src` + legacy import hook. |

All former top-level modules (`rag_service`, `retrieval_db`, `image_mcp_tools`, …) resolve via `radius_rag._legacy_bootstrap` (`MetaPathFinder`), installed from `conftest`, `rag_server.py`, and `radius_rag` package import.

| Layer | Purpose |
|-------|---------|
| `src/radius_rag/**` | Canonical implementation. |
| `tools/migrate_*` | Copy edited modules into `src/` (no root shims). |
| `tools/remove_root_shims.py` | Delete obsolete root `*.py` if reintroduced. |
| `data/` | Optional local overrides; default workflow in `src/radius_rag/imaging/data/`. |
