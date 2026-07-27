# AGENTS.md — Инструкции для разработки rag_server

## Обзор проекта

`rag_server/` — standalone RAG search server для индекса, собранного `rag_crawler`.

Возможности:

- FastAPI HTTP API и MCP tools (`search_project`, `ask_project`, `web_search`)
- OpenAI-совместимый `/v1/chat/completions` с RAG-контекстом
- Hybrid search: FAISS + BM25, optional rerank
- Admin API для загрузки, активации и rollback bundle

## Структура проекта

```
rag_server.py      — FastAPI app, MCP, admin endpoints
rag_service.py     — RagService: загрузка индекса, retrieve, ask
rag_bundle.py      — валидация, staging, activate/rollback
rag_search.py      — legacy search helper
rag_proxy.py       — proxy utilities
test_rag_bundle.py — тесты bundle и поиска
requirements.txt   — зависимости (дублирует pyproject.toml)
pyproject.toml     — сборка и dev-инструменты
```

## Контракт metadata: `source_type`

Значения приходят из crawler metadata:

| Значение | Описание |
|----------|----------|
| `code` | Исходный код |
| `documentation` | Markdown, DOCX и другие документы |

Фильтры MCP/API и boost (`RAG_SOURCE_TYPE_BOOSTS`) используют эти строки как есть.
`path_prefix` сопоставляется и с абсолютными путями metadata, и с относительными сегментами (`backend/sources/radius-ipc`).

## Переменные окружения

### Индекс (локальный fallback)

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `RAG_INDEX_FILE` | `rag_index.faiss` | Путь к FAISS index |
| `RAG_METADATA_FILE` | `rag_metadata.jsonl` | Путь к metadata |

### Admin API

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `RAG_ADMIN_TOKEN` | — | Bearer token для admin endpoints |
| `RAG_STAGING_DIR` | `rag_staging` | Каталог staged bundle |
| `RAG_BUNDLE_STATE_DIR` | `rag_bundle_state` | Маркеры active/previous |
| `RAG_UPLOAD_MAX_BYTES` | `536870912` | Лимит размера upload |

### Поиск

| Переменная | Назначение |
|------------|------------|
| `RAG_EMBEDDING_URL` | URL embedding API |
| `RAG_EMBEDDING_MODEL` | Имя модели embeddings |
| `RAG_RERANK_URL` | URL rerank API (optional) |
| `RAG_RERANK_ENABLED` | `true`/`false` |
| `RAG_SOURCE_TYPE_BOOSTS` | JSON, например `{"documentation": 1.5}` |

### LLM и web search

| Переменная | Назначение |
|------------|------------|
| `LLM_URL`, `LLM_MODEL` | Backend для `ask_project` |
| `SEARXNG_URL` | SearXNG для `web_search` |

## Admin API

Все endpoints требуют `Authorization: Bearer <RAG_ADMIN_TOKEN>`.

| Method | Path | Описание |
|--------|------|----------|
| POST | `/admin/index/upload` | Загрузить ZIP bundle в staging |
| POST | `/admin/index/activate/{staging_id}` | Активировать staged bundle |
| GET | `/admin/index/status` | Статус active/previous bundle |
| POST | `/admin/index/reload` | Перезагрузить active bundle |
| POST | `/admin/index/rollback` | Откатиться на previous bundle |

## Bundle contract

ZIP содержит ровно три файла:

- `rag_index.faiss`
- `rag_metadata.jsonl`
- `rag_manifest.json`

Manifest включает checksums, `document_count`, `dimensions`, `embedding_model`, `built_at`.

## Фильтры поиска

MCP tools и HTTP API принимают:

- `source_type` — `code` или `documentation`
- `language` — значение из metadata (`python`, `markdown`, …)
- `path_prefix` — префикс пути файла; поддерживаются абсолютные пути и относительные сегменты (`backend/sources/radius-ipc`)

В OpenAI chat completions фильтры передаются как `rag_source_type`, `rag_language`, `rag_path_prefix` в теле запроса.

## Запуск

```bash
pip install -e ".[dev]"

# или
pip install -r requirements.txt

uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

## Проверка

```bash
ruff check .
mypy rag_server.py rag_service.py rag_bundle.py
python -m pytest test_rag_bundle.py -v
```

На Windows при `PermissionError` в pytest задайте `TMPDIR` внутри workspace.

## Стиль

- UTF-8, LF line endings
- Минимальный diff, без несвязанного рефакторинга

## Связанный проект

`rag_crawler/` — pipeline сборки индекса. См. `rag_crawler/AGENTS.md`.
