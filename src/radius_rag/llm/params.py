#!/usr/bin/env python3

import os
from typing import Any


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


def env_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return float(value)


def env_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return int(value)


# Рекомендации Qwen/Qwen3.8-27B для non-thinking (instruct) режима.
LLM_TEMPERATURE = env_float(
    "RAG_LLM_TEMPERATURE",
    0.7,
)
LLM_TOP_P = env_float(
    "RAG_LLM_TOP_P",
    0.8,
)
LLM_TOP_K = env_int(
    "RAG_LLM_TOP_K",
    20,
)
LLM_PRESENCE_PENALTY = env_float(
    "RAG_LLM_PRESENCE_PENALTY",
    1.5,
)
LLM_REPETITION_PENALTY = env_float(
    "RAG_LLM_REPETITION_PENALTY",
    1.0,
)
LLM_ENABLE_THINKING = env_bool(
    "RAG_LLM_ENABLE_THINKING",
    False,
)
LLM_FORCE_DEFAULTS = env_bool(
    "RAG_LLM_FORCE_DEFAULTS",
    False,
)


def apply_llm_defaults(
    payload: dict[str, Any],
    *,
    override_thinking: bool | None = None,
) -> dict[str, Any]:
    """Подставляет sampling-параметры для LLM, если клиент их не передал.

    When ``override_thinking`` is set (e.g. safety validator), it always wins over
    ``RAG_LLM_FORCE_DEFAULTS`` and ``RAG_LLM_ENABLE_THINKING``.
    """
    result = dict(payload)
    force = LLM_FORCE_DEFAULTS

    def set_param(
        key: str,
        value: Any,
    ) -> None:
        if force or key not in result:
            result[key] = value

    set_param("temperature", LLM_TEMPERATURE)
    set_param("top_p", LLM_TOP_P)
    set_param("top_k", LLM_TOP_K)
    set_param("presence_penalty", LLM_PRESENCE_PENALTY)
    set_param("repetition_penalty", LLM_REPETITION_PENALTY)

    extra_body = dict(result.get("extra_body") or {})
    chat_template_kwargs = dict(
        extra_body.get("chat_template_kwargs") or {}
    )

    if override_thinking is not None:
        chat_template_kwargs["enable_thinking"] = override_thinking
    elif force or "enable_thinking" not in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = LLM_ENABLE_THINKING

    extra_body["chat_template_kwargs"] = chat_template_kwargs
    result["extra_body"] = extra_body

    return result
