#!/usr/bin/env python3

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


INDEX_FILE = Path(os.getenv("RAG_INDEX_FILE", "rag_index.faiss"))
METADATA_FILE = Path(
    os.getenv("RAG_METADATA_FILE", "rag_metadata.jsonl")
)

LLM_URL = os.getenv(
    "LLM_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)
EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://127.0.0.1:8002/v1/embeddings",
)

LLM_MODEL = os.getenv("LLM_MODEL", "rag-assistant")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding",
)

TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "24000")
)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))


index: faiss.Index | None = None
metadata: list[dict[str, Any]] = []


def load_metadata(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid metadata JSON at line {line_number}: {error}"
                ) from error

    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    global index, metadata

    if not INDEX_FILE.exists():
        raise RuntimeError(f"FAISS index not found: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise RuntimeError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))
    metadata = load_metadata(METADATA_FILE)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata mismatch: "
            f"{index.ntotal} != {len(metadata)}"
        )

    print(
        f"Loaded RAG index: documents={index.ntotal}, "
        f"dimensions={index.d}"
    )

    yield


app = FastAPI(
    title="Project RAG Proxy",
    version="1.0.0",
    lifespan=lifespan,
)


def extract_query(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    # Берём последние пользовательские сообщения.
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            parts.append(content)

        elif isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)

        if len(parts) >= 2:
            break

    return "\n".join(reversed(parts)).strip()


async def get_embedding(
    client: httpx.AsyncClient,
    text: str,
) -> np.ndarray:
    response = await client.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": [text],
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data")

    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError("Unexpected embedding response")

    vector = np.asarray(
        data[0]["embedding"],
        dtype=np.float32,
    ).reshape(1, -1)

    faiss.normalize_L2(vector)

    if index is None:
        raise RuntimeError("FAISS index is not loaded")

    if vector.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimensions mismatch: "
            f"{vector.shape[1]} != {index.d}"
        )

    return vector


def search_index(
    query_vector: np.ndarray,
) -> list[tuple[float, dict[str, Any]]]:
    if index is None:
        raise RuntimeError("FAISS index is not loaded")

    scores, indices = index.search(query_vector, TOP_K)

    results: list[tuple[float, dict[str, Any]]] = []

    for score, item_index in zip(scores[0], indices[0]):
        if item_index < 0:
            continue

        if item_index >= len(metadata):
            continue

        results.append(
            (
                float(score),
                metadata[item_index],
            )
        )

    return results


def build_context(
    results: list[tuple[float, dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    total_chars = 0

    for position, (score, item) in enumerate(results, start=1):
        file_path = str(item.get("file", ""))
        symbol = str(
            item.get("full_name")
            or item.get("name")
            or ""
        )
        detail = str(item.get("detail", ""))
        language = str(item.get("language", ""))
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        code = str(item.get("code", ""))

        chunk = "\n".join(
            [
                f"[Source {position}]",
                f"Score: {score:.6f}",
                f"Language: {language}",
                f"File: {file_path}",
                f"Symbol: {symbol}",
                f"Signature: {detail}",
                f"Lines: {start_line}-{end_line}",
                "Code:",
                code,
            ]
        )

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk[:remaining]

        chunks.append(chunk)
        total_chars += len(chunk)

        sources.append(
            {
                "score": score,
                "file": file_path,
                "symbol": symbol,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    return "\n\n---\n\n".join(chunks), sources


def enrich_messages(
    original_messages: list[dict[str, Any]],
    context: str,
) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    other_messages: list[dict[str, Any]] = []

    for message in original_messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append(content)
        else:
            other_messages.append(message)

    system_parts.append(
        (
            "You are a code assistant for the Project project.\n"
            "Use the retrieved project sources below when relevant.\n"
            "Do not invent project APIs, files, types, or behavior.\n"
            "If the supplied context is insufficient, say so explicitly.\n"
            "When referring to project code, mention the source file and symbol.\n\n"
            "Retrieved Project sources:\n\n"
            f"{context}"
        )
    )

    return [
        {
            "role": "system",
            "content": "\n\n".join(system_parts),
        },
        *other_messages,
    ]

async def upstream_stream(payload: dict[str, Any]):
    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            LLM_URL,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(
                    f"LLM returned {response.status_code}: "
                    f"{body.decode('utf-8', errors='replace')}"
                )

            async for chunk in response.aiter_raw():
                yield chunk

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents": index.ntotal if index is not None else 0,
        "dimensions": index.d if index is not None else 0,
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-assistant",
                "object": "model",
                "created": 0,
                "owned_by": "radius",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
            detail="'messages' must be a non-empty array",
        )

    query = extract_query(messages)

    if not query:
        raise HTTPException(
            status_code=400,
            detail="No user text found in messages",
        )

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            query_vector = await get_embedding(client, query)
            results = search_index(query_vector)
            context, sources = build_context(results)

            upstream_payload = dict(body)

            # Поле model клиента игнорируется.
            upstream_payload["model"] = LLM_MODEL
            upstream_payload["messages"] = enrich_messages(
                messages,
                context,
            )

            if bool(body.get("stream", False)):
                return StreamingResponse(
                    upstream_stream(upstream_payload),
                    media_type="text/event-stream",
                    headers={
                        "X-RAG-Sources": str(len(sources)),
                    },
                )

            response = await client.post(
                LLM_URL,
                json=upstream_payload,
            )
            response.raise_for_status()

            result = response.json()

            # Не ломает OpenAI-совместимый ответ:
            # дополнительное поле клиенты обычно игнорируют.
            result["rag_sources"] = sources

            return JSONResponse(result)

    except httpx.HTTPStatusError as error:
        body_text = error.response.text

        raise HTTPException(
            status_code=error.response.status_code,
            detail=body_text,
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

