#!/usr/bin/env python3

import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from rag_bundle import BundleStore, BundleValidationError, stage_bundle

from llm_params import apply_llm_defaults
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

mcp = FastMCP(
    name="Project Knowledge Gateway",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            "192.168.10.250:*",
            "192.168.13.109:*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "http://192.168.10.250:*",
            "192.168.13.109:*",
        ],
    ),
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
    else:
        load_bundle(active[0])

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
    expose_headers=["Mcp-Session-Id"],
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

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".zip",
            prefix="upload-",
            dir=staging_dir,
            delete=False,
        ) as archive:
            archive_path = Path(archive.name)
            received = 0

            async for chunk in request.stream():
                received += len(chunk)

                if received > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Bundle exceeds the configured size limit",
                    )

                archive.write(chunk)

        bundle_dir, manifest = stage_bundle(archive_path, staging_dir, max_size)

    except BundleValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)

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

    try:
        path, manifest = bundle_store().activate(staging_id)
        load_bundle(path)
    except BundleValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
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
    except BundleValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
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
        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding request failed: "
                f"{error}"
            ),
        ) from error

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=error.response.text,
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    upstream_payload = dict(body)

    upstream_payload["model"] = LLM_MODEL
    upstream_payload["messages"] = (
        enriched_messages
    )
    upstream_payload = apply_llm_defaults(upstream_payload)

    if bool(body.get("stream", False)):
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
            response = await client.post(
                LLM_URL,
                json=upstream_payload,
            )
            response.raise_for_status()

        result = response.json()
        result["rag_sources"] = sources

        return JSONResponse(result)

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=error.response.text,
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM request failed: "
                f"{error}"
            ),
        ) from error


mcp_app = mcp.streamable_http_app()
app.mount("/mcp", mcp_app)

