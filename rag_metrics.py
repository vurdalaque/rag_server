"""Prometheus metrics for rag_server (VictoriaMetrics / Grafana scrape)."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator, TypeVar

import httpx
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

F = TypeVar("F", bound=Callable[..., Any])

EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://127.0.0.1:8002/v1/embeddings",
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding",
)
LLM_URL = os.getenv(
    "LLM_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "rag-assistant",
)
RERANK_ENABLED = os.getenv(
    "RAG_RERANK_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}


def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


METRICS_ENABLED = env_bool(
    "RAG_METRICS_ENABLED",
    True,
)
METRICS_PATH = os.getenv(
    "RAG_METRICS_PATH",
    "/metrics",
)
PROBE_ENABLED = env_bool(
    "RAG_METRICS_PROBE_ENABLED",
    True,
)
PROBE_INTERVAL = float(
    os.getenv(
        "RAG_METRICS_PROBE_INTERVAL",
        "30",
    )
)

INDEX_LOADED = Gauge(
    "rag_index_loaded",
    "Whether a FAISS index is loaded (1/0).",
)
DOCUMENTS_TOTAL = Gauge(
    "rag_documents_total",
    "Number of documents in the loaded FAISS index.",
)
INDEX_DIMENSIONS = Gauge(
    "rag_index_dimensions",
    "Vector dimensions of the loaded FAISS index.",
)
RERANK_ENABLED_GAUGE = Gauge(
    "rag_rerank_enabled",
    "Whether reranking is enabled (1/0).",
)

DEPENDENCY_UP = Gauge(
    "rag_dependency_up",
    "Whether an upstream dependency responded successfully (1/0).",
    ["service"],
)
DEPENDENCY_REQUESTS = Counter(
    "rag_dependency_requests_total",
    "Upstream dependency requests during RAG operations.",
    ["service", "outcome"],
)
DEPENDENCY_ERRORS = Counter(
    "rag_dependency_errors_total",
    "Upstream dependency errors during RAG operations.",
    ["service", "error_type"],
)
DEPENDENCY_DURATION = Histogram(
    "rag_dependency_request_duration_seconds",
    "Upstream dependency request latency during RAG operations.",
    ["service"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

RETRIEVE_DURATION = Histogram(
    "rag_retrieve_duration_seconds",
    "End-to-end retrieve() latency.",
    ["has_filters"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
RETRIEVE_RESULTS = Histogram(
    "rag_retrieve_results",
    "Number of results returned by retrieve().",
    buckets=(0, 1, 2, 3, 6, 10, 20),
)
RETRIEVE_EMPTY = Counter(
    "rag_retrieve_empty_total",
    "Retrieve calls that returned no results.",
    ["reason"],
)
FAISS_DURATION = Histogram(
    "rag_faiss_search_duration_seconds",
    "FAISS search latency inside retrieve().",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
BM25_DURATION = Histogram(
    "rag_bm25_duration_seconds",
    "BM25 scoring latency inside retrieve().",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
RERANK_DURATION = Histogram(
    "rag_rerank_duration_seconds",
    "Reranker latency inside retrieve().",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
CONTEXT_CHARS = Histogram(
    "rag_context_chars",
    "Total characters in built RAG context.",
    buckets=(500, 1000, 2500, 5000, 10000, 20000, 40000, 80000),
)

MCP_TOOL_CALLS = Counter(
    "rag_mcp_tool_calls_total",
    "MCP tool invocations.",
    ["tool", "outcome"],
)
MCP_TOOL_DURATION = Histogram(
    "rag_mcp_tool_duration_seconds",
    "MCP tool invocation latency.",
    ["tool"],
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    ),
)
MCP_SEARCH_RESULTS = Histogram(
    "rag_mcp_search_results",
    "Result count returned by search-like MCP tools.",
    ["tool"],
    buckets=(0, 1, 2, 3, 6, 10, 20),
)

_IMAGE_GENERATION_STAGES = frozenset(
    {
        "probe",
        "validate",
        "safety",
        "comfy_execute",
        "generate",
    }
)
_IMAGE_GENERATION_OUTCOMES = frozenset(
    {
        "success",
        "error",
        "blocked",
        "skipped",
    }
)

IMAGE_GENERATION_EVENTS = Counter(
    "rag_image_generation_total",
    "Image generation stage outcomes.",
    ["stage", "outcome"],
)
IMAGE_GENERATION_DURATION = Histogram(
    "rag_image_generation_duration_seconds",
    "Image generation stage latency.",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

IMAGE_GENERATION_ENABLED = Gauge(
    "rag_image_generation_enabled",
    "Whether image MCP tools registered after startup probe (1/0).",
)

CHAT_COMPLETIONS = Counter(
    "rag_chat_completions_total",
    "OpenAI-compatible chat completion requests.",
    ["stream", "outcome"],
)
CHAT_RAG_SOURCES = Histogram(
    "rag_chat_rag_sources",
    "Number of RAG sources attached to chat completions.",
    buckets=(0, 1, 2, 3, 6, 10, 20),
)
CHAT_CONTEXT_INJECTED = Counter(
    "rag_chat_context_injected_total",
    "Chat completions where non-empty RAG context was built.",
)

BUNDLE_UPLOAD = Counter(
    "rag_bundle_upload_total",
    "Bundle upload attempts.",
    ["outcome"],
)
BUNDLE_UPLOAD_BYTES = Histogram(
    "rag_bundle_upload_bytes",
    "Uploaded bundle sizes in bytes.",
    buckets=(1e5, 5e5, 1e6, 5e6, 10e6, 50e6, 100e6, 250e6, 500e6),
)
BUNDLE_ACTIVATE = Counter(
    "rag_bundle_activate_total",
    "Bundle activate attempts.",
    ["outcome"],
)
BUNDLE_ACTIVATE_DURATION = Histogram(
    "rag_bundle_activate_duration_seconds",
    "Bundle activate latency.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
BUNDLE_ROLLBACK = Counter(
    "rag_bundle_rollback_total",
    "Bundle rollback attempts.",
    ["outcome"],
)

HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "HTTP requests handled by rag_server.",
    ["handler", "method", "status"],
)
HTTP_DURATION = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency.",
    ["handler", "method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

SERVER_INFO = Info(
    "rag_server",
    "Static rag_server build and configuration metadata.",
)


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def metrics_payload() -> bytes:
    return generate_latest()


def classify_error(
    error: BaseException,
) -> str:
    if isinstance(error, httpx.ConnectError):
        return "connect"

    if isinstance(error, httpx.TimeoutException):
        return "timeout"

    if isinstance(error, httpx.HTTPStatusError):
        return "http"

    return "other"


def normalize_handler_path(
    path: str,
) -> str:
    if path == METRICS_PATH:
        return "/metrics"

    if path.startswith("/admin/index/activate/"):
        return "/admin/index/activate/{staging_id}"

    if path.startswith("/mcp"):
        return "/mcp"

    return path


def update_index_state(
    service: Any,
) -> None:
    if not METRICS_ENABLED:
        return

    INDEX_LOADED.set(1 if service.index_loaded else 0)
    DOCUMENTS_TOTAL.set(service.document_count)
    INDEX_DIMENSIONS.set(service.dimensions)
    RERANK_ENABLED_GAUGE.set(1 if RERANK_ENABLED else 0)
    SERVER_INFO.info(
        {
            "embedding_url": EMBEDDING_URL,
            "embedding_model": EMBEDDING_MODEL,
            "llm_url": LLM_URL,
            "llm_model": LLM_MODEL,
        }
    )


@contextmanager
def track_dependency(
    service: str,
) -> Iterator[None]:
    if not METRICS_ENABLED:
        yield
        return

    start = time.perf_counter()
    outcome = "success"

    try:
        yield
    except BaseException as error:
        outcome = "error"
        DEPENDENCY_ERRORS.labels(
            service=service,
            error_type=classify_error(error),
        ).inc()
        raise
    finally:
        DEPENDENCY_REQUESTS.labels(
            service=service,
            outcome=outcome,
        ).inc()
        DEPENDENCY_DURATION.labels(service=service).observe(
            time.perf_counter() - start
        )


@contextmanager
def track_duration(
    histogram: Histogram,
    **labels: str,
) -> Iterator[None]:
    if not METRICS_ENABLED:
        yield
        return

    start = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start

        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


def record_retrieve_empty(
    reason: str,
) -> None:
    if METRICS_ENABLED:
        RETRIEVE_EMPTY.labels(reason=reason).inc()


def record_retrieve_result(
    count: int,
    has_filters: bool,
    duration_seconds: float,
) -> None:
    if not METRICS_ENABLED:
        return

    RETRIEVE_RESULTS.observe(count)
    RETRIEVE_DURATION.labels(
        has_filters="true" if has_filters else "false",
    ).observe(duration_seconds)


def record_context_chars(
    total_chars: int,
) -> None:
    if METRICS_ENABLED:
        CONTEXT_CHARS.observe(total_chars)


def record_chat_completion(
    stream: bool,
    outcome: str,
    sources_count: int,
    context_injected: bool,
) -> None:
    if not METRICS_ENABLED:
        return

    CHAT_COMPLETIONS.labels(
        stream="true" if stream else "false",
        outcome=outcome,
    ).inc()
    CHAT_RAG_SOURCES.observe(sources_count)

    if context_injected:
        CHAT_CONTEXT_INJECTED.inc()


def record_bundle_upload(
    outcome: str,
    size_bytes: int | None = None,
) -> None:
    if not METRICS_ENABLED:
        return

    BUNDLE_UPLOAD.labels(outcome=outcome).inc()

    if size_bytes is not None and outcome == "success":
        BUNDLE_UPLOAD_BYTES.observe(size_bytes)


def record_bundle_activate(
    outcome: str,
    duration_seconds: float | None = None,
) -> None:
    if not METRICS_ENABLED:
        return

    BUNDLE_ACTIVATE.labels(outcome=outcome).inc()

    if duration_seconds is not None and outcome == "success":
        BUNDLE_ACTIVATE_DURATION.observe(duration_seconds)


def record_bundle_rollback(
    outcome: str,
) -> None:
    if METRICS_ENABLED:
        BUNDLE_ROLLBACK.labels(outcome=outcome).inc()


def _bounded_image_label(
    value: str,
    allowed: frozenset[str],
    fallback: str,
) -> str:
    normalized = value.strip().lower()

    if normalized in allowed:
        return normalized

    return fallback


def record_image_generation(
    stage: str,
    outcome: str,
) -> None:
    if not METRICS_ENABLED:
        return

    IMAGE_GENERATION_EVENTS.labels(
        stage=_bounded_image_label(stage, _IMAGE_GENERATION_STAGES, "generate"),
        outcome=_bounded_image_label(
            outcome,
            _IMAGE_GENERATION_OUTCOMES,
            "error",
        ),
    ).inc()


@contextmanager
def track_image_stage(
    stage: str,
) -> Iterator[None]:
    bounded = _bounded_image_label(stage, _IMAGE_GENERATION_STAGES, "generate")

    with track_duration(IMAGE_GENERATION_DURATION, stage=bounded):
        yield


def track_mcp_tool(
    tool_name: str,
) -> Callable[[F], F]:
    def decorator(
        func: F,
    ) -> F:
        @wraps(func)
        async def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not METRICS_ENABLED:
                return await func(*args, **kwargs)

            start = time.perf_counter()
            outcome = "success"

            try:
                result = await func(*args, **kwargs)

                if isinstance(result, dict):
                    count = result.get("count")

                    if isinstance(count, int):
                        MCP_SEARCH_RESULTS.labels(tool=tool_name).observe(count)

                try:
                    from mcp.types import CallToolResult

                    if isinstance(result, CallToolResult) and result.is_error:
                        outcome = "error"
                except ImportError:
                    pass

                return result
            except Exception:
                outcome = "error"
                raise
            finally:
                MCP_TOOL_CALLS.labels(
                    tool=tool_name,
                    outcome=outcome,
                ).inc()
                MCP_TOOL_DURATION.labels(tool=tool_name).observe(
                    time.perf_counter() - start
                )

        return wrapper  # type: ignore[return-value]

    return decorator


async def probe_embedding(
    timeout: float,
) -> bool:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.post(
                EMBEDDING_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": ["metrics probe"],
                },
            )
            response.raise_for_status()
    except BaseException:
        return False

    return True


async def probe_searxng(
    searxng_url: str,
    timeout: float,
) -> bool:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            response = await client.get(
                searxng_url,
                params={
                    "q": "metrics probe",
                    "format": "json",
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "rag-server-metrics-probe/1.0",
                },
            )
            response.raise_for_status()
    except BaseException:
        return False

    return True


def _comfyui_base_url() -> str:
    return (
        os.getenv("COMFYUI_URL")
        or os.getenv("COMFYUI_BASE_URL")
        or "http://127.0.0.1:8188"
    ).rstrip("/")


async def probe_comfyui(
    timeout: float,
) -> bool:
    if not env_bool("IMAGE_GENERATION_ENABLED", False):
        return False

    url = f"{_comfyui_base_url()}/object_info"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except BaseException:
        return False

    return True


def set_image_generation_enabled(
    enabled: bool,
) -> None:
    if METRICS_ENABLED:
        IMAGE_GENERATION_ENABLED.set(1 if enabled else 0)


async def probe_llm(
    timeout: float,
) -> bool:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.post(
                LLM_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "ping",
                        }
                    ],
                    "max_tokens": 1,
                },
            )
            response.raise_for_status()
    except BaseException:
        return False

    return True


async def run_dependency_probes(
    searxng_url: str,
    probe_timeout: float,
) -> None:
    if not METRICS_ENABLED or not PROBE_ENABLED:
        return

    embedding_ok = await probe_embedding(probe_timeout)
    DEPENDENCY_UP.labels(service="embedding").set(1 if embedding_ok else 0)

    llm_ok = await probe_llm(probe_timeout)
    DEPENDENCY_UP.labels(service="llm").set(1 if llm_ok else 0)

    searxng_ok = await probe_searxng(searxng_url, probe_timeout)
    DEPENDENCY_UP.labels(service="searxng").set(1 if searxng_ok else 0)

    comfy_ok = await probe_comfyui(probe_timeout)
    if env_bool("IMAGE_GENERATION_ENABLED", False):
        DEPENDENCY_UP.labels(service="comfyui").set(1 if comfy_ok else 0)


@asynccontextmanager
async def dependency_probe_loop(
    searxng_url: str,
) -> AsyncIterator[None]:
    if not METRICS_ENABLED or not PROBE_ENABLED:
        yield
        return

    probe_timeout = min(PROBE_INTERVAL, 10.0)
    task = asyncio.create_task(
        _dependency_probe_worker(searxng_url, probe_timeout)
    )

    try:
        yield
    finally:
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass


async def _dependency_probe_worker(
    searxng_url: str,
    probe_timeout: float,
) -> None:
    while True:
        await run_dependency_probes(searxng_url, probe_timeout)
        await asyncio.sleep(PROBE_INTERVAL)
