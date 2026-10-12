"""Request-scoped context for retrieval observability."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Callable, Iterator

request_id_var: ContextVar[str | None] = ContextVar("retrieval_request_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("retrieval_correlation_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("retrieval_session_id", default=None)
request_source_var: ContextVar[str | None] = ContextVar("retrieval_request_source", default=None)
request_payload_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "retrieval_request_payload",
    default=None,
)

ResponsePayloadBuilder = Callable[[dict[str, Any]], dict[str, Any]]
_response_payload_builder_var: ContextVar[ResponsePayloadBuilder | None] = ContextVar(
    "retrieval_response_payload_builder",
    default=None,
)


def new_request_id() -> str:
    return str(uuid.uuid4())


def get_request_id() -> str | None:
    return request_id_var.get()


def get_correlation_id() -> str | None:
    return correlation_id_var.get()


def get_session_id() -> str | None:
    return session_id_var.get()


def get_request_source() -> str | None:
    return request_source_var.get()


def get_request_payload() -> dict[str, Any]:
    value = request_payload_var.get()
    return dict(value) if value is not None else {}


def set_response_payload_builder(
    builder: ResponsePayloadBuilder | None,
) -> Token[ResponsePayloadBuilder | None]:
    return _response_payload_builder_var.set(builder)


def reset_response_payload_builder(token: Token[ResponsePayloadBuilder | None]) -> None:
    _response_payload_builder_var.reset(token)


def build_response_payload(base: dict[str, Any]) -> dict[str, Any]:
    builder = _response_payload_builder_var.get()
    if builder is None:
        return dict(base)
    return builder(dict(base))


@contextmanager
def retrieval_context(
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
    session_id: str | None = None,
    request_source: str | None = None,
    request_payload: dict[str, Any] | None = None,
    response_payload_builder: ResponsePayloadBuilder | None = None,
) -> Iterator[None]:
    tokens: list[tuple[ContextVar[Any], Token[Any]]] = []

    def _set(var: ContextVar[Any], value: Any) -> None:
        tokens.append((var, var.set(value)))

    if request_id is not None:
        _set(request_id_var, request_id)
    if correlation_id is not None:
        _set(correlation_id_var, correlation_id)
    if session_id is not None:
        _set(session_id_var, session_id)
    if request_source is not None:
        _set(request_source_var, request_source)
    if request_payload is not None:
        _set(request_payload_var, dict(request_payload))
    if response_payload_builder is not None:
        _set(_response_payload_builder_var, response_payload_builder)

    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
