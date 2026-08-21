from __future__ import annotations

import importlib

import pytest

import llm_params


def reload_llm_params(
    monkeypatch: pytest.MonkeyPatch,
    **env: str,
) -> None:
    for name in (
        "RAG_LLM_TEMPERATURE",
        "RAG_LLM_TOP_P",
        "RAG_LLM_TOP_K",
        "RAG_LLM_PRESENCE_PENALTY",
        "RAG_LLM_REPETITION_PENALTY",
        "RAG_LLM_ENABLE_THINKING",
        "RAG_LLM_FORCE_DEFAULTS",
    ):
        monkeypatch.delenv(name, raising=False)

    for name, value in env.items():
        monkeypatch.setenv(name, value)

    importlib.reload(llm_params)


def test_apply_llm_defaults_uses_qwen_non_thinking_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_llm_params(monkeypatch)

    payload = llm_params.apply_llm_defaults(
        {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    assert payload["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }


def test_apply_llm_defaults_preserves_client_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_llm_params(monkeypatch)

    payload = llm_params.apply_llm_defaults(
        {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
            "top_p": 0.5,
        }
    )

    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.5
    assert payload["top_k"] == 20


def test_apply_llm_defaults_can_force_over_client_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_llm_params(monkeypatch, RAG_LLM_FORCE_DEFAULTS="true")

    payload = llm_params.apply_llm_defaults(
        {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
            "top_p": 0.5,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        }
    )

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
