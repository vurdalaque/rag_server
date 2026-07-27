# rag_server

RAG search server для индекса, собранного [rag_crawler](../rag_crawler/README.md).

Предоставляет hybrid search (FAISS + BM25), MCP tools, OpenAI-совместимый API и admin endpoints для доставки bundle.

## Возможности

- `GET /health` — статус, `index_loaded`, число документов
- `GET /debug/search` — отладочный поиск с фильтрами
- `POST /v1/chat/completions` — RAG-чат (OpenAI-совместимый)
- `POST /admin/index/*` — upload, activate, rollback bundle
- MCP: `search_project`, `ask_project`, `web_search`, `ping`

## Требования

- Python 3.10+
- Embedding API (тот же model/dimensions, что при сборке индекса)
- Опционально: LLM API для `ask_project`, SearXNG для `web_search`

## Установка

```bash
cd rag_server
pip install -e ".[dev]"

# или
pip install -r requirements.txt
```

## Быстрый старт

```powershell
# Windows
copy env.example .env
# отредактируйте RAG_ADMIN_TOKEN и EMBEDDING_URL

.\start.ps1
```

```bash
# Linux / macOS
cp env.example .env
chmod +x start.sh
./start.sh
```

Скрипт читает `.env`, создаёт `data/staging` и `data/state`, запускает uvicorn.

Сервер стартует **без индекса** — поиск будет пустым, пока не загрузите bundle через admin API или не укажете `RAG_INDEX_FILE` / `RAG_METADATA_FILE`.

Параметры:

```powershell
.\start.ps1 -Port 8010 -BindHost 127.0.0.1
.\start.ps1 -EnvFile .env.local
```

```bash
./start.sh --port 8010 --env-file .env.local
```

Проверка:

```bash
curl http://localhost:8000/health
```

## Запуск вручную

Если нужен запуск без скрипта:

```bash
export RAG_ADMIN_TOKEN=your-secret-token
export EMBEDDING_URL=http://192.168.10.250:8002/v1/embeddings
export EMBEDDING_MODEL=qwen3-embedding

# опционально: стартовый индекс до первого deploy
export RAG_INDEX_FILE=/path/to/rag_index.faiss
export RAG_METADATA_FILE=/path/to/rag_metadata.jsonl

export RAG_STAGING_DIR=./data/staging
export RAG_BUNDLE_STATE_DIR=./data/state

uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

При старте server загружает active bundle из `RAG_BUNDLE_STATE_DIR` или fallback-файлы из `RAG_INDEX_FILE` / `RAG_METADATA_FILE`.

## Деплой индекса

С машины сборки (rag_crawler):

```bash
export RAG_SERVER_URL=http://your-server:8000
export RAG_ADMIN_TOKEN=your-secret-token
python -m rag_crawler --config config.yaml deploy
```

Или вручную:

```bash
# 1. Upload ZIP (rag_index.faiss + rag_metadata.jsonl + rag_manifest.json)
curl -X POST http://localhost:8000/admin/index/upload \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/zip" \
  --data-binary @bundle.zip

# 2. Activate
curl -X POST http://localhost:8000/admin/index/activate/{staging_id} \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN"

# 3. Status
curl http://localhost:8000/admin/index/status \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN"

# 4. Rollback
curl -X POST http://localhost:8000/admin/index/rollback \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN"
```

## Поиск

### Debug API

```bash
curl "http://localhost:8000/debug/search?query=ipc&source_type=code&language=cpp&path_prefix=backend/sources/radius-ipc&top_k=5"
```

### Фильтры

| Параметр | Описание |
|----------|----------|
| `source_type` | `code` или `documentation` |
| `language` | `python`, `cpp`, `markdown`, … |
| `path_prefix` | Абсолютный префикс или относительный сегмент пути (`backend/sources/radius-ipc`) |

`path_prefix` работает и с абсолютными путями в metadata (`D:/projects/...`).

### OpenAI chat completions

В теле запроса можно передать:

- `rag_source_type`
- `rag_language`
- `rag_path_prefix`

## Переменные окружения

### Индекс

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `RAG_INDEX_FILE` | `rag_index.faiss` | Fallback index |
| `RAG_METADATA_FILE` | `rag_metadata.jsonl` | Fallback metadata |

### Admin

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `RAG_ADMIN_TOKEN` | — | Bearer token (обязателен для admin) |
| `RAG_STAGING_DIR` | `rag_staging` | Staging bundle (скрипт по умолчанию: `./data/staging`) |
| `RAG_BUNDLE_STATE_DIR` | `rag_bundle_state` | Active/previous markers (скрипт: `./data/state`) |
| `RAG_UPLOAD_MAX_BYTES` | `536870912` | Лимит upload (512 MB) |

### Поиск

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `EMBEDDING_URL` | `http://127.0.0.1:8002/v1/embeddings` | Embedding API |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Модель embeddings |
| `RERANK_URL` | — | Rerank API (optional) |
| `RAG_RERANK_ENABLED` | `true` | Включить rerank |
| `RAG_SOURCE_TYPE_BOOSTS` | `{}` | JSON boost, напр. `{"documentation": 1.5}` |

### LLM и web

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `LLM_URL` | `http://127.0.0.1:8001/v1/chat/completions` | LLM для ask |
| `LLM_MODEL` | `rag-assistant` | Имя модели |
| `SEARXNG_URL` | `http://127.0.0.1:8003/search` | Web search |

## Bundle contract

ZIP содержит ровно:

- `rag_index.faiss`
- `rag_metadata.jsonl`
- `rag_manifest.json`

Server валидирует checksums и соответствие `FAISS.ntotal` числу строк metadata до активации.

## Разработка

```bash
ruff check .
mypy rag_server.py rag_service.py rag_bundle.py
python -m pytest test_rag_bundle.py -v
```

На Windows при `PermissionError` в pytest:

```bash
set TMPDIR=D:\path\to\workspace\.tmp
```

Подробности — в [AGENTS.md](AGENTS.md).

## Связанные проекты

- [rag_crawler](../rag_crawler/README.md) — сборка и публикация индекса
