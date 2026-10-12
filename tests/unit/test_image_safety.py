from __future__ import annotations

import asyncio
import importlib
import json as json_mod
from typing import Any

import httpx
import pytest

import image_safety
import llm_client
import llm_params


def reload_safety_modules(
    monkeypatch: pytest.MonkeyPatch,
    **env: str,
) -> None:
    for name in (
        "IMAGE_GENERATION_SAFETY_ENABLED",
        "IMAGE_GENERATION_SAFETY_FAIL_CLOSED",
        "IMAGE_GENERATION_SAFETY_TIMEOUT",
        "RAG_LLM_ENABLE_THINKING",
        "RAG_LLM_FORCE_DEFAULTS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(llm_params)
    importlib.reload(llm_client)
    importlib.reload(image_safety)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def capture_llm_payload(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        assert json is not None
        captured.append(json)
        content = json_mod.dumps(
            {
                "allowed": True,
                "categories": [],
                "reason": "ok",
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return captured


def _image(mime: str = "image/png", data: bytes = b"\x89PNG") -> image_safety.SafetyImage:
    return image_safety.SafetyImage(mime_type=mime, data=data)


def _user_content_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload["messages"]
    user = next(m for m in messages if m["role"] == "user")
    content = user["content"]
    assert isinstance(content, list)
    return content


async def _would_submit_to_comfy(
    validator: image_safety.ImageSafetyValidator,
    prompt: str,
    images: list[image_safety.SafetyImage],
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    result = await validator.validate(prompt, images, client=client)
    return not validator.blocks_on_result(result)


def test_safety_disabled_skips_llm_call(
    monkeypatch: pytest.MonkeyPatch,
    capture_llm_payload: list[dict[str, Any]],
) -> None:
    reload_safety_modules(monkeypatch, IMAGE_GENERATION_SAFETY_ENABLED="false")
    validator = image_safety.ImageSafetyValidator(enabled=False)

    result = asyncio.run(validator.validate("draw a cat", [_image()]))

    assert result.status == "skipped"
    assert result.allowed is True
    assert capture_llm_payload == []


def test_safety_enabled_sends_prompt_and_all_images(
    monkeypatch: pytest.MonkeyPatch,
    capture_llm_payload: list[dict[str, Any]],
) -> None:
    reload_safety_modules(monkeypatch, IMAGE_GENERATION_SAFETY_ENABLED="true")
    validator = image_safety.ImageSafetyValidator(enabled=True)
    images = [_image(data=b"one"), _image(data=b"two")]

    asyncio.run(validator.validate("sunset over water", images))

    assert len(capture_llm_payload) == 1
    parts = _user_content_parts(capture_llm_payload[0])
    text_blob = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
    assert "sunset over water" in text_blob
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_safety_request_bypasses_rag(
    monkeypatch: pytest.MonkeyPatch,
    capture_llm_payload: list[dict[str, Any]],
) -> None:
    reload_safety_modules(monkeypatch, IMAGE_GENERATION_SAFETY_ENABLED="true")
    validator = image_safety.ImageSafetyValidator(enabled=True)

    asyncio.run(validator.validate("office diagram", [_image()]))

    payload = capture_llm_payload[0]
    system = next(m for m in payload["messages"] if m["role"] == "system")
    assert system["content"] == image_safety.SAFETY_SYSTEM_PROMPT
    serialized = json_mod.dumps(payload).lower()
    assert "ask_project" not in serialized
    assert "retriev" not in serialized
    assert "rag_context" not in serialized


def test_safety_request_forces_thinking_false(
    monkeypatch: pytest.MonkeyPatch,
    capture_llm_payload: list[dict[str, Any]],
) -> None:
    reload_safety_modules(
        monkeypatch,
        IMAGE_GENERATION_SAFETY_ENABLED="true",
        RAG_LLM_FORCE_DEFAULTS="true",
        RAG_LLM_ENABLE_THINKING="true",
    )
    validator = image_safety.ImageSafetyValidator(enabled=True)

    asyncio.run(validator.validate("test", []))

    kwargs = capture_llm_payload[0]["extra_body"]["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is False


def test_safety_allowed_allows_generation(
    monkeypatch: pytest.MonkeyPatch,
    capture_llm_payload: list[dict[str, Any]],
) -> None:
    reload_safety_modules(monkeypatch, IMAGE_GENERATION_SAFETY_ENABLED="true")
    validator = image_safety.ImageSafetyValidator(enabled=True)

    assert asyncio.run(_would_submit_to_comfy(validator, "flowers", [])) is True


def test_safety_rejected_blocks_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_safety_modules(monkeypatch, IMAGE_GENERATION_SAFETY_ENABLED="true")

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        body = json_mod.dumps(
            {
                "allowed": False,
                "categories": ["policy"],
                "reason": "blocked",
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"content": body}},
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    validator = image_safety.ImageSafetyValidator(enabled=True)

    assert asyncio.run(_would_submit_to_comfy(validator, "bad", [])) is False


def test_safety_validator_error_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_safety_modules(
        monkeypatch,
        IMAGE_GENERATION_SAFETY_ENABLED="true",
        IMAGE_GENERATION_SAFETY_FAIL_CLOSED="true",
    )

    async def broken_post(
        self: httpx.AsyncClient,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "post", broken_post)
    validator = image_safety.ImageSafetyValidator(enabled=True, fail_closed=True)

    assert asyncio.run(_would_submit_to_comfy(validator, "x", [])) is False


def test_safety_validator_error_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_safety_modules(
        monkeypatch,
        IMAGE_GENERATION_SAFETY_ENABLED="true",
        IMAGE_GENERATION_SAFETY_FAIL_CLOSED="false",
    )

    async def broken_post(
        self: httpx.AsyncClient,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "post", broken_post)
    validator = image_safety.ImageSafetyValidator(enabled=True, fail_closed=False)

    result = asyncio.run(validator.validate("x", []))
    assert result.status == "error"
    assert validator.blocks_on_result(result) is False
