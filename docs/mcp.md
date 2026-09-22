# MCP-сервер rag_server

`rag_server` поднимает MCP (Model Context Protocol) поверх того же индекса и `RagService`, что и HTTP API.

- **Имя сервера:** `Project Knowledge Gateway`
- **Версия:** `1.2.0`
- **Endpoint:** `POST /mcp/` (mount от корня FastAPI-приложения)
- **Транспорт:** Streamable HTTP, stateless, ответы в JSON (`json_response=true`)

Проверка доступности без MCP-сессии:

```bash
curl http://localhost:8000/health
```

В ответе поле `interfaces.mcp` и список `mcp_tools`.

---

## Протокол

### Заголовки запроса

| Заголовок | Обязательно | Значение |
|-----------|-------------|----------|
| `Content-Type` | да | `application/json` |
| `Accept` | да | `application/json, text/event-stream` |
| `MCP-Protocol-Version` | для modern flow | `2026-07-28` |
| `Mcp-Method` | опционально | например `server/discover` |

Если прокси срезает routing-заголовки, сервер умеет восстановить их из `params._meta` в JSON-теле (см. `wrap_mcp_modern_headers` в `rag_server.py`).

### Поддерживаемые версии

| Версия | Сценарий |
|--------|----------|
| `2024-11-05` | классический flow: `initialize` → `notifications/initialized` → `tools/list` |
| `2026-07-28` | modern flow для ChatGPT Connector: `server/discover` |

Все запросы и ответы — **JSON-RPC 2.0**:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "...",
  "params": { }
}
```

Успешный ответ:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { }
}
```

### Modern discover (`server/discover`)

Запрос:

```http
POST /mcp/
Accept: application/json, text/event-stream
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: server/discover
```

```json
{
  "jsonrpc": "2.0",
  "id": "discover-1",
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "chatgpt",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Ответ (`result`):

| Поле | Тип | Описание |
|------|-----|----------|
| `supportedVersions` | `string[]` | `["2026-07-28"]` |
| `capabilities` | `object` | `tools`, `resources`, `prompts`, … |
| `cacheScope` | `string` | `"public"` |
| `ttlMs` | `number` | TTL кэша discover/list, 3_600_000 |
| `_meta.io.modelcontextprotocol/serverInfo` | `object` | `name`, `version` сервера |

### Legacy initialize

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": { "name": "test", "version": "1.0" }
  }
}
```

Затем notification (без `id`):

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
```

### Список инструментов (`tools/list`)

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

В `result.tools` четыре инструмента:

| Имя | Назначение |
|-----|------------|
| `search_project` | Семантический поиск по индексу (основной) |

> **Draft — image generation (conditional):** при `IMAGE_GENERATION_ENABLED=true` и успешном probe ComfyUI в `tools/list` появляются `generate_image` и `image_generation_capabilities`. Иначе список совпадает с базовым (см. `/health` → `mcp_tools`).
| `ask_project` | Поиск + ответ через upstream LLM |
| `web_search` | Поиск в интернете через SearXNG |
| `ping` | Проверка доступности MCP |

### Вызов инструмента (`tools/call`)

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search_project",
    "arguments": {
      "query": "Radius IPC authentication",
      "top_k": 6,
      "repo": "radius",
      "source_type": "code"
    }
  }
}
```

Результат инструмента MCP SDK упаковывает в `result.content[]` (обычно `type: "text"` с JSON-строкой). Ниже описана **логическая структура** dict, который возвращают Python-функции инструментов.

---

## Инструменты

### `search_project`

Поиск фрагментов в FAISS+BM25 индексе. Не вызывает LLM.

**Параметры**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `query` | `string` | — | Поисковый запрос (обязательный) |
| `top_k` | `integer` | `6` | Число результатов (1…`MAX_TOP_K`) |
| `source_type` | `string?` | `null` | `code` или `documentation` |
| `language` | `string?` | `null` | `python`, `cpp`, `markdown`, … |
| `path_prefix` | `string?` | `null` | Префикс относительного пути в repo |
| `repo` | `string?` | `null` | Логическое имя из `source_roots` (`radius`, `radius-doc`, …) |

**Ответ (успех, есть результаты)**

```json
{
  "query": "Radius IPC",
  "searched_sources": ["Project source-code index"],
  "count": 2,
  "retry_recommended": false,
  "message": "Relevant Project source-code fragments were found.",
  "results": [ /* SearchHit[] */ ]
}
```

**Ответ (индекс не загружен)**

```json
{
  "query": "...",
  "searched_sources": [],
  "count": 0,
  "results": [],
  "retry_recommended": false,
  "message": "Search index is not loaded. Upload and activate a bundle via /admin/index/*."
}
```

**Ответ (пустая выдача)**

```json
{
  "query": "...",
  "searched_sources": ["Project source-code index"],
  "count": 0,
  "results": [],
  "retry_recommended": false,
  "message": "No relevant Project source-code fragments were found. Do not repeat the same search using minor paraphrases."
}
```

#### Объект `SearchHit` (`results[]`)

Каждый элемент — один чанк из `rag_metadata.jsonl` после ранжирования.

| Поле | Тип | Описание |
|------|-----|----------|
| `metadata_index` | `integer` | Индекс записи в metadata / FAISS |
| `score` | `number` | Итоговая оценка для сортировки |
| `hybrid_score` | `number?` | Комбинация FAISS+BM25 rank (если rerank выключен — равен `score`) |
| `faiss_score` | `number?` | Cosine similarity embedding-запроса |
| `bm25_score` | `number?` | BM25 по тексту чанка |
| `rerank_score` | `number?` | Оценка rerank-модели (если `RAG_RERANK_ENABLED=true`) |
| `recency_factor` | `number?` | Множитель свежести 0.25…1.0 (если есть `updated_at`) |
| `repo` | `string` | Логическое имя репозитория |
| `file` | `string` | Относительный путь файла в repo |
| `source_type` | `string` | `code` или `documentation` |
| `language` | `string` | Язык / формат (`python`, `markdown`, …) |
| `symbol` | `string` | Имя символа / секции |
| `signature` | `string` | Сигнатура / заголовок |
| `start_line` | `integer?` | Начальная строка |
| `end_line` | `integer?` | Конечная строка |
| `git_url` | `string` | Ссылка на источник (git blob или Confluence URL) |
| `updated_at` | `string?` | ISO-дата последнего изменения (Confluence dump) |
| `code` | `string` | Текст фрагмента, попавший в индекс |

**Ранжирование**

1. Кандидаты из FAISS и BM25 объединяются.
2. `hybrid_score = rank_score × source_type_boost × recency_factor`
3. При включённом rerank финальный `score = rerank_score × source_type_boost × recency_factor`

Переменные окружения: `RAG_SOURCE_TYPE_BOOSTS`, `RAG_RECENCY_*` (см. `env.example`).

---

### `ask_project`

Retrieval + вызов upstream LLM (`LLM_URL`). Удобен, если у клиента нет своей модели.

**Параметры** — те же фильтры, что у `search_project`, плюс `question` вместо `query`.

| Параметр | Тип | По умолчанию |
|----------|-----|--------------|
| `question` | `string` | — |
| `top_k` | `integer` | `6` |
| `source_type` | `string?` | `null` |
| `language` | `string?` | `null` |
| `path_prefix` | `string?` | `null` |
| `repo` | `string?` | `null` |

**Ответ**

```json
{
  "answer": "Текст ответа от LLM…",
  "sources": [
    {
      "score": 0.0123,
      "rerank_score": null,
      "faiss_score": 0.87,
      "repo": "radius-doc",
      "file": "fusion/AGENTS.md",
      "source_type": "documentation",
      "symbol": "Overview",
      "start_line": 1,
      "end_line": 42,
      "git_url": "https://gitlab.example.com/…"
    }
  ]
}
```

`sources` — укороченная версия hit-ов (без поля `code`). Полный текст фрагментов уходит в prompt LLM через `build_context`.

---

### `generate_image` (draft, conditional)

Генерация изображений через ComfyUI. Инструмент **не регистрируется**, если при старте probe не прошёл.

**Параметры** (model-agnostic):

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `prompt` | `string` | — | Текстовый prompt |
| `negative_prompt` | `string?` | `null` | Negative prompt |
| `width` | `integer?` | server default | Ширина (px) |
| `height` | `integer?` | server default | Высота (px) |
| `steps` | `integer?` | server default | Шаги сэмплера |
| `seed` | `integer?` | `null` | Seed (опционально) |
| `cfg` | `number?` | server default | CFG scale |
| `sampler` | `string?` | server default | Имя sampler |
| `scheduler` | `string?` | server default | Scheduler |
| `image_count` | `integer` | `1` | Количество изображений (≤ policy) |
| `reference_images` | `string[]?` | `null` | Base64 reference images |

**Ответ:** MCP `CallToolResult` — блоки `image` (base64) в `content`; метаданные (`seed`, `prompt_id`, `image_count`, `timings`) в `structuredContent` + `outputSchema` в discovery. `image_generation_capabilities` возвращает structured JSON с лимитами и `samplers`.

**Ошибки:** JSON `{"error": {"code", "message", "details?"}}` — например `backend_unavailable`, `safety_blocked`, `invalid_prompt`.

Переменные: `IMAGE_GENERATION_*`, `COMFYUI_*` в `env.example`.

---

### `image_generation_capabilities` (draft, conditional)

Возвращает лимиты сервера (`max_images`, `max_output_bytes`, reference limits), defaults (resolution, steps, cfg, sampler) и при доступности ComfyUI — enum sampler/scheduler из `/object_info`.

---

### `web_search`

Поиск в интернете через локальный SearXNG (`SEARXNG_URL`). Не использует RAG-индекс.

**Параметры**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `query` | `string` | — | Поисковый запрос |
| `limit` | `integer` | `5` | Макс. результатов |
| `language` | `string` | `"all"` | Язык SearXNG |
| `categories` | `string` | `"general"` | Категории SearXNG |

**Ответ (успех)**

```json
{
  "query": "OpenCV face detection",
  "searched_sources": ["public Internet via SearXNG"],
  "count": 3,
  "retry_recommended": false,
  "message": "Web search results were found.",
  "results": [
    {
      "title": "…",
      "url": "https://…",
      "snippet": "…",
      "engine": "google",
      "engines": ["google", "bing"],
      "score": 1.23,
      "category": "general",
      "published_date": "2025-01-15T00:00:00"
    }
  ]
}
```

При ошибке SearXNG инструмент бросает исключение → JSON-RPC error в MCP-ответе.

---

### `ping`

Без параметров.

**Ответ**

```json
{
  "status": "ok",
  "server": "Project Knowledge Gateway",
  "tools": ["search_project", "web_search", "ask_project"]
}
```

---

## Ресурсы MCP

| URI | Описание |
|-----|----------|
| `project://status` | JSON: `status`, `index_loaded`, `documents`, `dimensions`, `searxng_url` |

Читается через MCP `resources/read`, не через HTTP GET.

---

## Соответствие HTTP API

| MCP tool | HTTP-аналог |
|----------|-------------|
| `search_project` | `GET /debug/search?query=…` |
| `ask_project` | нет прямого REST; ближе к `/v1/chat/completions` с RAG |
| `web_search` | только MCP |
| `ping` | `GET /health` (частично) |

`/debug/search` возвращает `{ "query", "count", "results" }` без обёртки `message` / `searched_sources`, но массив `results` имеет тот же формат `SearchHit`.

---

## Примеры

### Discover (ChatGPT Connector)

```bash
curl -s http://localhost:8000/mcp/ \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -d '{
    "jsonrpc":"2.0",
    "id":"discover-1",
    "method":"server/discover",
    "params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}
  }' | jq .
```

### Поиск по коду

```bash
curl -s http://localhost:8000/mcp/ \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc":"2.0",
    "id":10,
    "method":"tools/call",
    "params":{
      "name":"search_project",
      "arguments":{
        "query":"SecurOS Radius integration",
        "top_k":5,
        "repo":"radius-doc",
        "source_type":"documentation"
      }
    }
  }' | jq .
```

### Эквивалент через debug API

```bash
curl -s "http://localhost:8000/debug/search?query=SecurOS+Radius&repo=radius-doc&source_type=documentation&top_k=5" | jq .
```

---

## Подключение клиентов

### Cursor / IDE

В `mcp.json` (или настройках MCP) указать URL сервера:

```json
{
  "mcpServers": {
    "radius-rag": {
      "url": "http://192.168.10.250:8000/mcp/"
    }
  }
}
```

### ChatGPT / Connector

Используется `server/discover` с protocol version `2026-07-28`. Сервер отдаёт публичный cache scope и capabilities, совпадающие с `initialize`.

---

## Ограничения и заметки

- MCP и `/debug/search` **не требуют** `RAG_ADMIN_TOKEN` (в отличие от `/admin/index/*`).
- Без активного bundle `search_project` вернёт `count: 0` и сообщение про незагруженный индекс.
- `path_prefix` сопоставляется с относительным `file` в metadata (POSIX slashes).
- Описания инструментов в MCP (`description` в `tools/list`) на английском — это system prompt для вызывающей модели; фактические данные всегда в JSON-ответе.
- Transport security: разрешённые host/origin заданы в `MCP_TRANSPORT_SECURITY` (`rag_server.py`); при деплое на другой хост список нужно расширить.

---

## Связанные файлы

| Файл | Содержание |
|------|------------|
| `rag_server.py` | MCP tools, mount `/mcp`, discover handler |
| `rag_service.py` | `retrieve`, `ask`, ранжирование, `build_context` |
| `test_mcp_discover.py` | Тесты discover и tools/list |
| `test_image_mcp_image.py` | Условная регистрация image MCP tools |
| `image_generation.py` | Backend facade, ComfyUI probe/generate |
| `image_mcp_tools.py` | `register_image_tools`, списки для `/health` / `ping` |
| `env.example` | `RAG_RECENCY_*`, `RAG_SOURCE_TYPE_BOOSTS`, LLM, SearXNG, `IMAGE_GENERATION_*` |
| `AGENTS.md` | Контракт `source_type`, admin API |
