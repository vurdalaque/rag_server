#!/usr/bin/env python3

import json
import os
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import DiscoverResult, RequestParams, ServerCapabilities, ToolsCapability
from mcp_types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_bundle import BundleStore, BundleValidationError, stage_bundle

from llm_params import apply_llm_defaults
from rag_metrics import (
    HTTP_DURATION,
    HTTP_REQUESTS,
    METRICS_ENABLED,
    METRICS_PATH,
    dependency_probe_loop,
    metrics_content_type,
    metrics_payload,
    normalize_handler_path,
    record_bundle_activate,
    record_bundle_rollback,
    record_bundle_upload,
    record_chat_completion,
    track_dependency,
    track_mcp_tool,
    update_index_state,
)
from rag_service import (
    LLM_MODEL,
    LLM_URL,
    REQUEST_TIMEOUT,
    rag_service,
)


SEARXNG_URL = os.getenv(
    "SEARXNG_URL",
    "http://127.0.0.1:8003/search",
)

WEB_SEARCH_TIMEOUT = float(
    os.getenv("WEB_SEARCH_TIMEOUT", "30"),
)

WEB_SEARCH_MAX_RESULTS = int(
    os.getenv("WEB_SEARCH_MAX_RESULTS", "10"),
)



def bundle_store() -> BundleStore:
    return BundleStore(
        Path(os.getenv("RAG_STAGING_DIR", "rag_staging")),
        Path(os.getenv("RAG_BUNDLE_STATE_DIR", "rag_bundle_state")),
    )


def require_admin(request: Request) -> None:
    token = os.getenv("RAG_ADMIN_TOKEN")

    if not token:
        raise HTTPException(
            status_code=503,
            detail="RAG_ADMIN_TOKEN is not configured",
        )

    authorization = request.headers.get("authorization", "")

    if not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def load_bundle(path: Path) -> None:
    rag_service.load_from_paths(
        path / "rag_index.faiss",
        path / "rag_metadata.jsonl",
    )

MCP_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "localhost:*",
        "127.0.0.1:*",
        "192.168.10.250:*",
        "192.168.13.250:*",
    ],
    allowed_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://192.168.10.250:*",
        "http://192.168.13.250:*",
    ],
)

mcp = MCPServer(
    name="Project Knowledge Gateway",
    version="1.2.0",
    cache_hints={
        "server/discover": CacheHint(scope="public", ttl_ms=3_600_000),
        "tools/list": CacheHint(scope="public", ttl_ms=3_600_000),
    },
)


def extract_query(
    messages: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            parts.append(content)

        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") != "text":
                    continue

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        if len(parts) >= 2:
            break

    return "\n".join(reversed(parts)).strip()


@mcp.tool()
@track_mcp_tool("search_project")
async def search_project(
    query: str,
    top_k: int = 6,
    source_type: str | None = None,
    language: str | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Search the indexed Project project source code.

    Use this tool for questions about Project implementation,
    classes, functions, symbols, files, internal APIs,
    database access and project architecture.

    The current index contains source code only.
    It does not contain public Internet information,
    README files, analyst documentation or API documentation.

    Do not repeat equivalent searches when no relevant results are found.
    """
    results = await rag_service.retrieve(
        query=query,
        top_k=top_k,
        source_type=source_type,
        language=language,
        path_prefix=path_prefix,
    )

    if not rag_service.index_loaded:
        return {
            "query": query,
            "searched_sources": [],
            "count": 0,
            "results": [],
            "retry_recommended": False,
            "message": (
                "Search index is not loaded. "
                "Upload and activate a bundle via /admin/index/*."
            ),
        }

    if not results:
        return {
            "query": query,
            "searched_sources": [
                "Project source-code index",
            ],
            "count": 0,
            "results": [],
            "retry_recommended": False,
            "message": (
                "No relevant Project source-code fragments were found. "
                "Do not repeat the same search using minor paraphrases."
            ),
        }

    return {
        "query": query,
        "searched_sources": [
            "Project source-code index",
        ],
        "count": len(results),
        "results": results,
        "retry_recommended": False,
        "message": (
            "Relevant Project source-code fragments were found."
        ),
    }

@mcp.tool()
@track_mcp_tool("web_search")
async def web_search(
    query: str,
    limit: int = 5,
    language: str = "all",
    categories: str = "general",
) -> dict[str, Any]:
    """
    Search the public Internet through the local SearXNG instance.

    Use this tool for:
    - current or recent information;
    - public documentation;
    - libraries, frameworks and external APIs;
    - general technical questions;
    - information that is not specific to the Project codebase.

    Do not use this tool to search Project implementation details.
    For Project source code use search_project instead.

    Do not repeat equivalent searches when the returned results are empty.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty")

    limit = max(
        1,
        min(limit, WEB_SEARCH_MAX_RESULTS),
    )

    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)

    headers = {
        "Accept": "application/json",
        "User-Agent": "Project-MCP-WebSearch/1.0",
    }

    params = {
        "q": query,
        "format": "json",
        "language": language,
        "categories": categories,
        "safesearch": 0,
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            with track_dependency("searxng"):
                response = await client.get(
                    SEARXNG_URL,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "SearXNG returned HTTP "
            f"{error.response.status_code}: "
            f"{error.response.text[:500]}"
        ) from error

    except httpx.RequestError as error:
        raise RuntimeError(
            f"SearXNG request failed: {error}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            "SearXNG returned a non-JSON response. "
            "Ensure that JSON format is enabled in settings.yml."
        ) from error

    raw_results = payload.get("results", [])

    if not isinstance(raw_results, list):
        raise RuntimeError(
            "Unexpected SearXNG response: "
            "'results' is not an array"
        )

    results: list[dict[str, Any]] = []

    for item in raw_results[:limit]:
        if not isinstance(item, dict):
            continue

        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": (
                    item.get("content")
                    or item.get("snippet")
                    or ""
                ),
                "engine": item.get("engine", ""),
                "engines": item.get("engines", []),
                "score": item.get("score"),
                "category": item.get("category", ""),
                "published_date": (
                    item.get("publishedDate")
                    or item.get("published_date")
                ),
            }
        )

    if not results:
        return {
            "query": query,
            "searched_sources": [
                "public Internet via SearXNG",
            ],
            "count": 0,
            "results": [],
            "retry_recommended": False,
            "message": (
                "No web search results were found. "
                "Do not repeat the same search using minor paraphrases."
            ),
        }

    return {
        "query": query,
        "searched_sources": [
            "public Internet via SearXNG",
        ],
        "count": len(results),
        "results": results,
        "retry_recommended": False,
        "message": "Web search results were found.",
    }


@mcp.tool()
@track_mcp_tool("ask_project")
async def ask_project(
    question: str,
    top_k: int = 6,
    source_type: str | None = None,
    language: str | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Answer a question about the Project project using
    semantic retrieval and the Project code model.

    Prefer search_project when the calling client already has
    access to the Project language model and can produce the answer itself.
    """
    return await rag_service.ask(
        question=question,
        top_k=top_k,
        source_type=source_type,
        language=language,
        path_prefix=path_prefix,
    )



@mcp.tool()
@track_mcp_tool("ping")
async def ping() -> dict[str, Any]:
    """
    Check that the Project MCP server is reachable.
    """
    return {
        "status": "ok",
        "server": "Project Knowledge Gateway",
        "tools": [
            "search_project",
            "web_search",
            "ask_project",
        ],
    }


@mcp.resource("project://status")
async def radius_status() -> str:
    return json.dumps(
        {
            "status": "ok",
            "index_loaded": rag_service.index_loaded,
            "documents": rag_service.document_count,
            "dimensions": rag_service.dimensions,
            "searxng_url": SEARXNG_URL,
        },
        ensure_ascii=False,
    )


async def upstream_stream(
    payload: dict[str, Any],
) -> AsyncIterator[bytes]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:
        async with client.stream(
            "POST",
            LLM_URL,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()

                raise RuntimeError(
                    "LLM returned "
                    f"{response.status_code}: "
                    + body.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            async for chunk in response.aiter_raw():
                yield chunk


@asynccontextmanager
async def lifespan(_: FastAPI):
    active = bundle_store().active_bundle()

    if active is None:
        if not rag_service.load_if_available():
            print(
                "RAG index not loaded; upload and activate a bundle via /admin/index/*"
            )
            update_index_state(rag_service)
    else:
        load_bundle(active[0])

    async with dependency_probe_loop(SEARXNG_URL):
        async with mcp.session_manager.run():
            yield


app = FastAPI(
    title="Project Knowledge Gateway",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "Mcp-Session-Id",
        "MCP-Protocol-Version",
        "Mcp-Method",
        "Mcp-Name",
    ],
)


@app.middleware("http")
async def record_http_metrics(
    request: Request,
    call_next,
):
    if not METRICS_ENABLED:
        return await call_next(request)

    handler = normalize_handler_path(request.url.path)

    if handler == METRICS_PATH:
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    HTTP_REQUESTS.labels(
        handler=handler,
        method=request.method,
        status=str(response.status_code),
    ).inc()
    HTTP_DURATION.labels(
        handler=handler,
        method=request.method,
    ).observe(duration)

    return response


if METRICS_ENABLED:
    @app.get(METRICS_PATH)
    async def metrics() -> Response:
        return Response(
            content=metrics_payload(),
            media_type=metrics_content_type(),
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_loaded": rag_service.index_loaded,
        "documents": rag_service.document_count,
        "dimensions": rag_service.dimensions,
        "llm_model": LLM_MODEL,
        "searxng_url": SEARXNG_URL,
        "interfaces": {
            "openai": "/v1/chat/completions",
            "mcp": "/mcp/",
        },
        "mcp_tools": [
            "search_project",
            "web_search",
            "ask_project",
            "ping",
        ],
    }


@app.post("/admin/index/upload")
async def upload_index(request: Request) -> dict[str, Any]:
    require_admin(request)


    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()

    if content_type != "application/zip":
        raise HTTPException(
            status_code=415,
            detail="Bundle content type must be application/zip",
        )

    try:
        max_size = int(os.getenv("RAG_UPLOAD_MAX_BYTES", str(512 * 1024 * 1024)))
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail="RAG_UPLOAD_MAX_BYTES is invalid",
        ) from error

    if max_size <= 0:
        raise HTTPException(
            status_code=500,
            detail="RAG_UPLOAD_MAX_BYTES must be positive",
        )

    content_length = request.headers.get("content-length")

    if content_length is not None:
        try:
            if int(content_length) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail="Bundle exceeds the configured size limit",
                )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="Invalid Content-Length",
            ) from error

    staging_dir = Path(os.getenv("RAG_STAGING_DIR", "rag_staging"))
    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    received = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".zip",
            prefix="upload-",
            dir=staging_dir,
            delete=False,
        ) as archive:
            archive_path = Path(archive.name)

            async for chunk in request.stream():
                received += len(chunk)

                if received > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Bundle exceeds the configured size limit",
                    )

                archive.write(chunk)

        bundle_dir, manifest = stage_bundle(archive_path, staging_dir, max_size)

    except HTTPException:
        record_bundle_upload("error")
        raise
    except BundleValidationError as error:
        record_bundle_upload("error")
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)

    record_bundle_upload("success", received)

    return {
        "status": "staged",
        "staging_id": bundle_dir.name,
        "document_count": manifest["document_count"],
        "dimensions": manifest["dimensions"],
    }



@app.get("/admin/index/status")
async def index_status(request: Request) -> dict[str, Any]:
    require_admin(request)

    try:
        return {
            "status": "ok",
            "index_loaded": rag_service.index_loaded,
            "documents": rag_service.document_count,
            "dimensions": rag_service.dimensions,
            **bundle_store().status(),
        }
    except BundleValidationError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/admin/index/activate/{staging_id}")
async def activate_index(staging_id: str, request: Request) -> dict[str, Any]:
    require_admin(request)
    start = time.perf_counter()

    try:
        path, manifest = bundle_store().activate(staging_id)
        load_bundle(path)
        record_bundle_activate("success", time.perf_counter() - start)
    except BundleValidationError as error:
        record_bundle_activate("error")
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        record_bundle_activate("error")
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "status": "active",
        "staging_id": staging_id,
        "document_count": manifest["document_count"],
        "dimensions": manifest["dimensions"],
    }


@app.post("/admin/index/reload")
async def reload_index(request: Request) -> dict[str, Any]:
    require_admin(request)

    try:
        active = bundle_store().active_bundle()

        if active is None:
            raise BundleValidationError("No active bundle is configured")

        path, manifest = active
        load_bundle(path)
    except BundleValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "status": "reloaded",
        "staging_id": path.name,
        "document_count": manifest["document_count"],
        "dimensions": manifest["dimensions"],
    }


@app.post("/admin/index/rollback")
async def rollback_index(request: Request) -> dict[str, Any]:
    require_admin(request)

    try:
        path, manifest = bundle_store().rollback()
        load_bundle(path)
        record_bundle_rollback("success")
    except BundleValidationError as error:
        record_bundle_rollback("error")
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        record_bundle_rollback("error")
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "status": "rolled_back",
        "staging_id": path.name,
        "document_count": manifest["document_count"],
        "dimensions": manifest["dimensions"],
    }

@app.get("/health/searxng")
async def health_searxng() -> dict[str, Any]:
    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                SEARXNG_URL,
                params={
                    "q": "test",
                    "format": "json",
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "Project-MCP-WebSearch/1.0"
                    ),
                },
            )
            response.raise_for_status()

        payload = response.json()

        return {
            "status": "ok",
            "url": SEARXNG_URL,
            "results": len(
                payload.get("results", [])
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"SearXNG check failed: {error}",
        ) from error


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "radius-code",
                "object": "model",
                "created": 0,
                "owned_by": "radius",
            }
        ],
    }


@app.get("/debug/search")
async def debug_search(
    query: str,
    top_k: int = 6,
    source_type: str | None = None,
    language: str | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    results = await rag_service.retrieve(
        query=query,
        top_k=top_k,
        source_type=source_type,
        language=language,
        path_prefix=path_prefix,
    )

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }



@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
):
    try:
        body = await request.json()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON: {error}",
        ) from error

    messages = body.get("messages")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            status_code=400,
            detail=(
                "'messages' must be a non-empty array"
            ),
        )

    query = extract_query(messages)

    if not query:
        raise HTTPException(
            status_code=400,
            detail="No user text found in messages",
        )

    requested_top_k = body.pop(
        "rag_top_k",
        None,
    )

    source_type = body.pop("rag_source_type", None)
    language = body.pop("rag_language", None)
    path_prefix = body.pop("rag_path_prefix", None)
    rag_use_system_prompt = body.pop(
        "rag_use_system_prompt",
        True,
    )

    if not isinstance(
        rag_use_system_prompt,
        bool,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "'rag_use_system_prompt' must be "
                "a boolean"
            ),
        )


    stream = bool(body.get("stream", False))

    try:
        enriched_messages, sources = (
            await rag_service.prepare_messages(
                messages=messages,
                query=query,
                top_k=requested_top_k,
                source_type=source_type,
                language=language,
                path_prefix=path_prefix,
                use_system_prompt=rag_use_system_prompt,

            )
        )

    except httpx.RequestError as error:
        record_chat_completion(
            stream=stream,
            outcome="error",
            sources_count=0,
            context_injected=False,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding request failed: "
                f"{error}"
            ),
        ) from error

    except httpx.HTTPStatusError as error:
        record_chat_completion(
            stream=stream,
            outcome="error",
            sources_count=0,
            context_injected=False,
        )
        raise HTTPException(
            status_code=error.response.status_code,
            detail=error.response.text,
        ) from error

    except Exception as error:
        record_chat_completion(
            stream=stream,
            outcome="error",
            sources_count=0,
            context_injected=False,
        )
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    context_injected = bool(sources)
    upstream_payload = dict(body)

    upstream_payload["model"] = LLM_MODEL
    upstream_payload["messages"] = (
        enriched_messages
    )
    upstream_payload = apply_llm_defaults(upstream_payload)

    if stream:
        record_chat_completion(
            stream=True,
            outcome="success",
            sources_count=len(sources),
            context_injected=context_injected,
        )
        return StreamingResponse(
            upstream_stream(upstream_payload),
            media_type="text/event-stream",
            headers={
                "X-RAG-Sources": str(
                    len(sources)
                ),
            },
        )

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
        ) as client:
            with track_dependency("llm"):
                response = await client.post(
                    LLM_URL,
                    json=upstream_payload,
                )
                response.raise_for_status()

        result = response.json()
        result["rag_sources"] = sources
        record_chat_completion(
            stream=False,
            outcome="success",
            sources_count=len(sources),
            context_injected=context_injected,
        )

        return JSONResponse(result)

    except httpx.HTTPStatusError as error:
        record_chat_completion(
            stream=False,
            outcome="error",
            sources_count=len(sources),
            context_injected=context_injected,
        )
        raise HTTPException(
            status_code=error.response.status_code,
            detail=error.response.text,
        ) from error

    except httpx.RequestError as error:
        record_chat_completion(
            stream=False,
            outcome="error",
            sources_count=len(sources),
            context_injected=context_injected,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM request failed: "
                f"{error}"
            ),
        ) from error


MCP_PROTOCOL_VERSION_META = "io.modelcontextprotocol/protocolVersion"
MCP_CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"
MCP_DEFAULT_MODERN_VERSION = LATEST_MODERN_VERSION


async def _connector_discover(
    _ctx: Any,
    _params: RequestParams | None,
) -> DiscoverResult:
    """ChatGPT-oriented discover: public cache, dual-era versions, tools only."""
    return DiscoverResult(
        supported_versions=[MCP_DEFAULT_MODERN_VERSION, LATEST_HANDSHAKE_VERSION],
        capabilities=ServerCapabilities(
            tools=ToolsCapability(list_changed=False),
        ),
        result_type="complete",
        cache_scope="public",
        ttl_ms=3_600_000,
    )


mcp._lowlevel_server.add_request_handler(
    "server/discover",
    RequestParams,
    _connector_discover,
)


def _normalize_modern_mcp_payload(
    payload: dict[str, Any],
    scope: Scope,
) -> bool:
    method = payload.get("method")
    if not isinstance(method, str):
        return False

    header_values = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    has_modern_header = "mcp-protocol-version" in header_values
    if method != "server/discover" and not has_modern_header:
        return False

    protocol_version = header_values.get(
        "mcp-protocol-version",
        MCP_DEFAULT_MODERN_VERSION,
    )

    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
        payload["params"] = params

    meta = params.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        params["_meta"] = meta

    changed = False
    if meta.get(MCP_PROTOCOL_VERSION_META) != protocol_version:
        meta[MCP_PROTOCOL_VERSION_META] = protocol_version
        changed = True
    if MCP_CLIENT_CAPABILITIES_META not in meta:
        meta[MCP_CLIENT_CAPABILITIES_META] = {}
        changed = True

    return changed


def _upsert_scope_header(scope: Scope, name: str, value: str) -> None:
    name_bytes = name.lower().encode("latin-1")
    value_bytes = value.encode("latin-1")
    headers = [
        (key, item)
        for key, item in scope.get("headers", [])
        if key.lower() != name_bytes
    ]
    headers.append((name_bytes, value_bytes))
    scope["headers"] = headers


async def _read_request_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message: Message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


class _BodyReplayReceive:
    def __init__(self, receive: Receive, body: bytes) -> None:
        self._receive = receive
        self._body = body
        self._done = False

    async def __call__(self) -> Message:
        if not self._done:
            self._done = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        return await self._receive()


def wrap_mcp_modern_headers(app: ASGIApp) -> ASGIApp:
    """Inject MCP routing headers from JSON _meta when proxies strip them."""

    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await app(scope, receive, send)
            return

        body = await _read_request_body(receive)
        replay_receive = _BodyReplayReceive(receive, body)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            await app(scope, replay_receive, send)
            return

        if not isinstance(payload, dict):
            await app(scope, replay_receive, send)
            return

        if _normalize_modern_mcp_payload(payload, scope):
            body = json.dumps(payload).encode("utf-8")
            replay_receive = _BodyReplayReceive(receive, body)

        params = payload.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        method = payload.get("method")

        if isinstance(meta, dict):
            protocol_version = meta.get(MCP_PROTOCOL_VERSION_META)
            if isinstance(protocol_version, str):
                _upsert_scope_header(scope, "mcp-protocol-version", protocol_version)
            if isinstance(method, str):
                _upsert_scope_header(scope, "mcp-method", method)

        await app(scope, replay_receive, send)

    return middleware


mcp_app = wrap_mcp_modern_headers(
    mcp.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        transport_security=MCP_TRANSPORT_SECURITY,
    )
)
app.mount("/mcp", mcp_app)

