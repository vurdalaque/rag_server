#!/usr/bin/env python3
"""Low-level async chat transport to the configured LLM (no RAG).

Environment:
    LLM_URL — OpenAI-compatible chat completions URL (same as ask_project).
    LLM_MODEL — model name sent in the JSON body.
    IMAGE_GENERATION_SAFETY_TIMEOUT — seconds for safety validation HTTP calls
        (falls back to REQUEST_TIMEOUT, then 120).
    REQUEST_TIMEOUT — shared default HTTP timeout when safety timeout unset.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from llm_params import apply_llm_defaults

LLM_URL = os.getenv(
    "LLM_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "rag-assistant",
)

_REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
SAFETY_LLM_TIMEOUT = float(
    os.getenv(
        "IMAGE_GENERATION_SAFETY_TIMEOUT",
        str(_REQUEST_TIMEOUT),
    )
)


def build_chat_payload(
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if thinking is not None:
        return apply_llm_defaults(payload, override_thinking=thinking)
    return apply_llm_defaults(payload)


def extract_message_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("LLM response content is not a string")
    return content


async def multimodal_chat(
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None = False,
    timeout: float | None = None,
    response_format: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """POST chat completions; default ``thinking=False`` for safety-style calls."""
    payload = build_chat_payload(
        messages,
        thinking=thinking,
        response_format=response_format,
    )
    http_timeout = httpx.Timeout(timeout if timeout is not None else SAFETY_LLM_TIMEOUT)

    if client is not None:
        response = await client.post(LLM_URL, json=payload)
        response.raise_for_status()
        return extract_message_content(response.json())

    async with httpx.AsyncClient(timeout=http_timeout) as owned:
        response = await owned.post(LLM_URL, json=payload)
        response.raise_for_status()
        return extract_message_content(response.json())
