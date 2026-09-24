"""Unit tests for multimodal image analysis (mocked LLM)."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, patch

import pytest

from image_analysis import (
    ImageAnalysisConfig,
    LlmImageAnalyzer,
    build_analysis_messages,
)
from image_generation_errors import BackendUnavailableError, InvalidRequestError
from test_image_contract import _png_b64


def _analysis_config(**overrides: object) -> ImageAnalysisConfig:
    base: dict[str, object] = {
        "enabled": True,
        "max_images": 3,
        "max_bytes": 512 * 1024,
        "timeout": 30.0,
        "probe_timeout": 5.0,
        "allowed_mime_types": frozenset({"image/png", "image/jpeg"}),
    }
    base.update(overrides)
    return ImageAnalysisConfig(**base)  # type: ignore[arg-type]


def test_build_analysis_messages_default_instruction() -> None:
    from image_analysis import AnalysisImage, DEFAULT_ANALYSIS_INSTRUCTION

    png = base64.b64decode(_png_b64())
    images = (AnalysisImage(mime_type="image/png", data=png),)
    messages = build_analysis_messages(images, None)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert user[0]["text"] == DEFAULT_ANALYSIS_INSTRUCTION
    assert any(part.get("type") == "image_url" for part in user)


def test_llm_analyzer_strips_multimodal_response() -> None:
    png = base64.b64decode(_png_b64())
    analyzer = LlmImageAnalyzer(config=_analysis_config())

    with patch(
        "image_analysis.multimodal_chat",
        new_callable=AsyncMock,
        return_value="  red sphere on gray background  ",
    ) as chat:
        result = asyncio.run(
            analyzer.analyze((png,), ("image/png",), "Describe the scene."),
        )

    assert result.text == "red sphere on gray background"
    assert "llm" in result.timings
    chat.assert_awaited_once()
    messages = chat.await_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_llm_analyzer_rejects_blank_instruction() -> None:
    png = base64.b64decode(_png_b64())
    analyzer = LlmImageAnalyzer(config=_analysis_config())

    with patch("image_analysis.multimodal_chat", new_callable=AsyncMock):
        with pytest.raises(InvalidRequestError):
            asyncio.run(analyzer.analyze((png,), ("image/png",), "   "))


def test_llm_analyzer_disabled_raises_backend_unavailable() -> None:
    png = base64.b64decode(_png_b64())
    analyzer = LlmImageAnalyzer(config=_analysis_config(enabled=False))

    with pytest.raises(BackendUnavailableError):
        asyncio.run(analyzer.analyze((png,), ("image/png",), None))
