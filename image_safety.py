#!/usr/bin/env python3
"""Optional multimodal pre-generation safety check via the configured LLM.

Does not use ask_project, retrieval, or RAG system prompts.

Environment:
    IMAGE_GENERATION_SAFETY_ENABLED — global on/off (default false).
    IMAGE_GENERATION_SAFETY_FAIL_CLOSED — when true (default), validator
        infrastructure errors block generation; when false, errors may allow
        generation to continue (still logged as error outcome).
    IMAGE_GENERATION_SAFETY_TIMEOUT — HTTP timeout for safety LLM calls
        (see llm_client.SAFETY_LLM_TIMEOUT).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from llm_client import multimodal_chat
from llm_params import env_bool

logger = logging.getLogger(__name__)

SAFETY_ENABLED = env_bool("IMAGE_GENERATION_SAFETY_ENABLED", False)
SAFETY_FAIL_CLOSED = env_bool("IMAGE_GENERATION_SAFETY_FAIL_CLOSED", True)

SafetyStatus = Literal["allowed", "rejected", "error", "skipped"]

SAFETY_SYSTEM_PROMPT = """You are a content safety classifier for image generation requests.
Evaluate the user's text prompt and all attached reference images together.
Respond with JSON only, no markdown, using this schema:
{"allowed": boolean, "categories": string[], "reason": string}
Set allowed to false if the request should be blocked for policy reasons.
categories lists violation types when blocked (empty when allowed).
reason is a short explanation. Do not include sensitive image data in reason."""


@dataclass(frozen=True)
class SafetyImage:
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class SafetyValidationResult:
    status: SafetyStatus
    allowed: bool
    categories: list[str] = field(default_factory=list)
    reason: str = ""


def _parse_validator_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("safety JSON root must be an object")
    return parsed


def _result_from_parsed(parsed: dict[str, Any]) -> SafetyValidationResult:
    allowed = parsed.get("allowed")
    if not isinstance(allowed, bool):
        raise ValueError("safety JSON missing boolean allowed")
    categories_raw = parsed.get("categories", [])
    if not isinstance(categories_raw, list):
        raise ValueError("safety JSON categories must be a list")
    categories = [str(item) for item in categories_raw]
    reason = str(parsed.get("reason", ""))
    if allowed:
        return SafetyValidationResult(
            status="allowed",
            allowed=True,
            categories=categories,
            reason=reason,
        )
    return SafetyValidationResult(
        status="rejected",
        allowed=False,
        categories=categories,
        reason=reason,
    )


def build_safety_messages(
    prompt: str,
    images: list[SafetyImage],
) -> list[dict[str, Any]]:
    user_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Evaluate this image generation request.\n\nPrompt:\n{prompt}",
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
                "text": f"(reference image {index})",
            }
        )

    return [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]


class ImageSafetyValidator:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        fail_closed: bool | None = None,
    ) -> None:
        self._enabled = SAFETY_ENABLED if enabled is None else enabled
        self._fail_closed = (
            SAFETY_FAIL_CLOSED if fail_closed is None else fail_closed
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def validate(
        self,
        prompt: str,
        images: list[SafetyImage],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> SafetyValidationResult:
        if not self._enabled:
            return SafetyValidationResult(
                status="skipped",
                allowed=True,
                reason="safety validation disabled",
            )

        messages = build_safety_messages(prompt, images)
        try:
            raw = await multimodal_chat(
                messages,
                thinking=False,
                response_format={"type": "json_object"},
                client=client,
            )
            return _result_from_parsed(_parse_validator_json(raw))
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("image safety validator error: %s", exc.__class__.__name__)
            return SafetyValidationResult(
                status="error",
                allowed=False,
                categories=["validator_error"],
                reason="safety validator unavailable",
            )

    def blocks_on_result(self, result: SafetyValidationResult) -> bool:
        if result.status == "rejected":
            return True
        if result.status == "error":
            return self._fail_closed
        return False
