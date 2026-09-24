#!/usr/bin/env python3
"""Multimodal image analysis via the configured LLM (no RAG).

Environment:
    IMAGE_ANALYSIS_ENABLED — global on/off (default true).
    IMAGE_ANALYSIS_MAX_IMAGES — max images per request (default 10).
    IMAGE_ANALYSIS_MAX_BYTES — max decoded bytes per image (default 10 MiB).
    IMAGE_ANALYSIS_TIMEOUT — HTTP timeout for analysis LLM calls (default REQUEST_TIMEOUT).
    IMAGE_ANALYSIS_PROBE_TIMEOUT — probe timeout (default 15).
    IMAGE_ANALYSIS_ALLOWED_MIME_TYPES — comma-separated input MIME types.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

import httpx

from image_generation_errors import (
    AnalysisFailedError,
    BackendUnavailableError,
    ImageGenerationError,
    InputTooLargeError,
    InvalidRequestError,
    TooManyImagesError,
    UnsupportedMimeTypeError,
)
from llm_client import LLM_URL, multimodal_chat
from llm_params import env_bool, env_float, env_int
from rag_metrics import probe_llm

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))

ANALYSIS_ENABLED = env_bool("IMAGE_ANALYSIS_ENABLED", True)
ANALYSIS_MAX_IMAGES = max(1, env_int("IMAGE_ANALYSIS_MAX_IMAGES", 10))
ANALYSIS_MAX_BYTES = max(
    1,
    env_int("IMAGE_ANALYSIS_MAX_BYTES", 10 * 1024 * 1024),
)
ANALYSIS_TIMEOUT = env_float("IMAGE_ANALYSIS_TIMEOUT", _REQUEST_TIMEOUT)
ANALYSIS_PROBE_TIMEOUT = env_float("IMAGE_ANALYSIS_PROBE_TIMEOUT", 15.0)

_MIME_RAW = os.getenv(
    "IMAGE_ANALYSIS_ALLOWED_MIME_TYPES",
    "image/png,image/jpeg,image/webp",
)
ANALYSIS_ALLOWED_MIME_TYPES = frozenset(
    part.strip().lower()
    for part in _MIME_RAW.split(",")
    if part.strip()
)

ANALYSIS_SYSTEM_PROMPT = """You are a visual analysis assistant.
Examine the image or images supplied in the user message and respond with clear,
accurate natural language. Follow the user's instruction when one is provided.
Base your answer only on what is visible in the images.
When multiple images are attached, refer to them in order as image 1, image 2, and so on.
Do not output JSON unless the user explicitly asks for structured data."""

DEFAULT_ANALYSIS_INSTRUCTION = (
    "Describe the image(s) in detail, including subjects, composition, colors, "
    "lighting, style, and any visible text."
)


@dataclass(frozen=True)
class AnalysisImage:
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    timings: dict[str, float]


@dataclass(frozen=True)
class ImageAnalysisConfig:
    enabled: bool
    max_images: int
    max_bytes: int
    timeout: float
    probe_timeout: float
    allowed_mime_types: frozenset[str]


def load_image_analysis_config() -> ImageAnalysisConfig:
    return ImageAnalysisConfig(
        enabled=ANALYSIS_ENABLED,
        max_images=ANALYSIS_MAX_IMAGES,
        max_bytes=ANALYSIS_MAX_BYTES,
        timeout=ANALYSIS_TIMEOUT,
        probe_timeout=ANALYSIS_PROBE_TIMEOUT,
        allowed_mime_types=ANALYSIS_ALLOWED_MIME_TYPES,
    )


def _normalize_mime(mime_type: str) -> str:
    return mime_type.strip().lower().split(";", 1)[0].strip()


def _validate_analysis_image(
    data: bytes,
    mime_type: str,
    *,
    index: int,
    config: ImageAnalysisConfig,
) -> None:
    normalized_mime = _normalize_mime(mime_type)
    if normalized_mime not in config.allowed_mime_types:
        raise UnsupportedMimeTypeError(
            f"image {index} has unsupported MIME type",
            index=index,
            mime_type=normalized_mime,
            allowed=sorted(config.allowed_mime_types),
        )

    if len(data) > config.max_bytes:
        raise InputTooLargeError(
            f"image {index} exceeds maximum size ({config.max_bytes} bytes)",
            index=index,
            size_bytes=len(data),
            max_bytes=config.max_bytes,
        )

    if not data:
        raise InvalidRequestError(
            f"image {index} is empty",
            index=index,
        )

    try:
        from PIL import Image
    except ImportError as error:
        raise InvalidRequestError(
            "server cannot validate analysis images (Pillow not installed)",
            index=index,
        ) from error

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except OSError as error:
        raise InvalidRequestError(
            f"image {index} is corrupt or truncated",
            index=index,
            reason=str(error),
            size_bytes=len(data),
        ) from error
    except Exception as error:
        raise InvalidRequestError(
            f"image {index} could not be decoded",
            index=index,
            reason=str(error),
            size_bytes=len(data),
        ) from error

    if width < 1 or height < 1:
        raise InvalidRequestError(
            f"image {index} has invalid dimensions",
            index=index,
            width=width,
            height=height,
        )


def _coerce_analysis_images(
    images: tuple[bytes, ...],
    mime_types: tuple[str, ...],
    *,
    config: ImageAnalysisConfig,
) -> tuple[AnalysisImage, ...]:
    if len(images) != len(mime_types):
        raise InvalidRequestError(
            "images and mime_types must have the same length",
            image_count=len(images),
            mime_count=len(mime_types),
        )

    if not images:
        raise InvalidRequestError("at least one image is required")

    if len(images) > config.max_images:
        raise TooManyImagesError(
            f"at most {config.max_images} images are allowed",
            count=len(images),
            max_images=config.max_images,
        )

    out: list[AnalysisImage] = []
    for index, (data, mime_type) in enumerate(zip(images, mime_types, strict=True), start=1):
        _validate_analysis_image(data, mime_type, index=index, config=config)
        out.append(
            AnalysisImage(
                mime_type=_normalize_mime(mime_type),
                data=data,
            )
        )
    return tuple(out)


def build_analysis_messages(
    images: tuple[AnalysisImage, ...],
    instruction: str | None,
) -> list[dict[str, Any]]:
    user_text = instruction if instruction is not None else DEFAULT_ANALYSIS_INSTRUCTION
    user_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_text,
        },
    ]
    for index, image in enumerate(images, start=1):
        encoded = base64.b64encode(image.data).decode("ascii")
        user_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.mime_type};base64,{encoded}",
                },
            }
        )
        user_parts.append(
            {
                "type": "text",
                "text": f"(image {index})",
            }
        )

    return [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]


@runtime_checkable
class ImageAnalyzer(Protocol):
    async def analyze(
        self,
        images: tuple[bytes, ...],
        mime_types: tuple[str, ...],
        instruction: str | None,
    ) -> AnalysisResult:
        """Run multimodal analysis on the supplied images."""


class LlmImageAnalyzer:
    def __init__(
        self,
        config: ImageAnalysisConfig | None = None,
    ) -> None:
        self._config = config or load_image_analysis_config()

    @property
    def config(self) -> ImageAnalysisConfig:
        return self._config

    async def probe(self) -> bool:
        if not self._config.enabled:
            return False
        return await probe_llm(self._config.probe_timeout)

    async def analyze(
        self,
        images: tuple[bytes, ...],
        mime_types: tuple[str, ...],
        instruction: str | None,
    ) -> AnalysisResult:
        if not self._config.enabled:
            raise BackendUnavailableError("image analysis is disabled")

        if instruction is not None:
            trimmed = instruction.strip()
            if not trimmed:
                raise InvalidRequestError("instruction must be non-empty when provided")
            if len(trimmed) > 8000:
                raise InvalidRequestError(
                    "instruction exceeds maximum length (8000 characters)",
                )
            instruction = trimmed

        analysis_images = _coerce_analysis_images(
            images,
            mime_types,
            config=self._config,
        )
        messages = build_analysis_messages(analysis_images, instruction)

        start = time.perf_counter()
        try:
            text = await multimodal_chat(
                messages,
                thinking=False,
                timeout=self._config.timeout,
            )
        except httpx.HTTPError as error:
            logger.warning("image analysis LLM HTTP error: %s", error.__class__.__name__)
            raise BackendUnavailableError(
                "image analysis backend is unavailable",
                reason=error.__class__.__name__,
            ) from error
        except (ValueError, TypeError) as error:
            logger.warning("image analysis LLM response error: %s", error)
            raise AnalysisFailedError(
                "image analysis failed",
                reason=str(error),
            ) from error
        except ImageGenerationError:
            raise
        except Exception as error:
            logger.warning("image analysis unexpected error: %s", error)
            raise AnalysisFailedError(
                "image analysis failed",
                reason=error.__class__.__name__,
            ) from error

        elapsed = time.perf_counter() - start
        cleaned = text.strip()
        if not cleaned:
            raise AnalysisFailedError("image analysis returned empty text")

        return AnalysisResult(
            text=cleaned,
            timings={"llm": elapsed},
        )


async def probe() -> bool:
    """Cheap readiness check for the analysis LLM (no inference)."""
    config = load_image_analysis_config()
    if not config.enabled:
        return False

    if await probe_llm(config.probe_timeout):
        return True

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.probe_timeout),
        ) as client:
            response = await client.get(LLM_URL)
            response.raise_for_status()
    except BaseException:
        return False

    return True
