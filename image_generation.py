"""Image generation backend facade (ComfyUI orchestration)."""

from __future__ import annotations

import logging
import mimetypes
import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import comfy_workflow
from comfy_client import ComfyUIClient
from image_generation_config import ImageGenerationConfig, load_image_generation_config
from image_generation_errors import (
    ImageGenerationError,
    InvalidRequestError,
    OutputTooLargeError,
    SafetyRejectedError,
    TooManyImagesError,
)
from image_reference import (
    canonicalize_reference_png_for_comfy,
    reference_sha256,
)
from image_safety import ImageSafetyValidator, SafetyImage
from rag_metrics import record_image_generation, track_image_stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime_type: str
    filename: str | None = None


@dataclass(frozen=True)
class ImageGenerationResult:
    images: list[GeneratedImage]
    seed: int | None
    prompt_id: str | None
    timings: dict[str, float]


@dataclass(frozen=True)
class GenerateImageRequest:
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    image_count: int = 1
    reference_images: tuple[bytes, ...] = ()
    mask: bytes | None = None
    sketch: bytes | None = None


@runtime_checkable
class ImageGenerationBackend(Protocol):
    async def probe(self) -> bool:
        """Check upstream readiness without running diffusion."""

    async def generate(
        self,
        request: GenerateImageRequest,
    ) -> ImageGenerationResult:
        """Validate input and run generation."""

    async def capabilities(self) -> dict[str, Any]:
        """Return model-agnostic limits and sampler metadata."""


def _guess_mime(data: bytes, index: int) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image/webp"
    guessed, _ = mimetypes.guess_type(f"ref{index}.bin")
    return guessed or "application/octet-stream"


class ComfyUIBackend:
    """Orchestrates config, ComfyUI client, workflow builder, and optional safety."""

    def __init__(
        self,
        config: ImageGenerationConfig | None = None,
        *,
        client: ComfyUIClient | None = None,
        safety: ImageSafetyValidator | None = None,
    ) -> None:
        self._config = config or load_image_generation_config()
        self._client = client
        self._safety = safety
        self._object_info: dict[str, Any] | None = None
        self._capabilities_cache: dict[str, Any] | None = None

    def _client_instance(self) -> ComfyUIClient:
        if self._client is not None:
            return self._client

        return ComfyUIClient(self._config)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _safety_validator(self) -> ImageSafetyValidator:
        if self._safety is not None:
            return self._safety

        return ImageSafetyValidator(
            enabled=self._config.safety_enabled,
            fail_closed=self._config.safety_fail_closed,
        )

    async def probe(self) -> bool:
        if not self._config.enabled:
            record_image_generation("probe", "skipped")
            return False

        with track_image_stage("probe"):
            try:
                comfy_workflow.validate_template_path(
                    self._config.workflow_template_path,
                )
                client = self._client_instance()
                object_info = await client.probe()
                comfy_workflow.validate_object_info(object_info)
                self._object_info = object_info
            except ImageGenerationError as error:
                logger.warning("Image generation probe failed: %s", error.message)
                record_image_generation("probe", "error")
                return False
            except Exception as error:
                logger.warning("Image generation probe failed: %s", error)
                record_image_generation("probe", "error")
                return False

            record_image_generation("probe", "success")
            root = self._config.comfyui_output_root.resolve()
            if str(os.getenv("COMFYUI_OUTPUT_ROOT", "")).strip() == "":
                logger.warning(
                    "COMFYUI_OUTPUT_ROOT is unset; reading ComfyUI files from %s "
                    "(set COMFYUI_OUTPUT_ROOT to ComfyUI's output directory)",
                    root,
                )
            else:
                logger.info("image generation probe ok comfy_output_root=%s", root)
            return True

    def _validate_request(
        self,
        request: GenerateImageRequest,
    ) -> GenerateImageRequest:
        prompt = request.prompt.strip()

        if not prompt:
            raise InvalidRequestError("prompt must be a non-empty string")

        if len(prompt) > 8000:
            raise InvalidRequestError("prompt exceeds maximum length (8000 characters)")

        if request.image_count < 1:
            raise InvalidRequestError("image_count must be at least 1")

        if request.image_count > self._config.max_images:
            raise InvalidRequestError(
                f"image_count exceeds server limit ({self._config.max_images})",
                image_count=request.image_count,
                max_images=self._config.max_images,
            )

        if len(request.reference_images) > self._config.max_reference_images:
            raise TooManyImagesError(
                f"At most {self._config.max_reference_images} reference images are allowed",
                count=len(request.reference_images),
                max_images=self._config.max_reference_images,
            )

        negative = request.negative_prompt

        if negative is not None and len(negative) > 8000:
            raise InvalidRequestError(
                "negative_prompt exceeds maximum length (8000 characters)",
            )

        record_image_generation("validate", "success")

        return GenerateImageRequest(
            prompt=prompt,
            negative_prompt=negative,
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=request.seed,
            cfg=request.cfg,
            sampler=request.sampler,
            scheduler=request.scheduler,
            image_count=request.image_count,
            reference_images=request.reference_images,
            mask=request.mask,
            sketch=request.sketch,
        )

    async def _upload_png_input(
        self,
        blob: bytes,
        *,
        upload_name: str,
        label: str,
        index: int | None = None,
    ) -> str:
        client = self._client_instance()
        digest_before = reference_sha256(blob)
        logger.info(
            "image input ok kind=%s index=%s bytes=%s sha256=%s",
            label,
            index,
            len(blob),
            digest_before[:16],
        )

        if "image/png" not in self._config.allowed_input_mime_types:
            raise InvalidRequestError(
                "input image mime type is not allowed",
                mime_type="image/png",
                allowed=sorted(self._config.allowed_input_mime_types),
            )

        if len(blob) > self._config.max_reference_bytes:
            details: dict[str, Any] = {
                "max_bytes": self._config.max_reference_bytes,
                "kind": label,
            }
            if index is not None:
                details["index"] = index
            raise InvalidRequestError(
                "input image exceeds size limit",
                **details,
            )

        uploaded = await client.upload_image(upload_name, blob, "image/png")
        await client.verify_uploaded_image_bytes(uploaded, blob, sha256_before=digest_before)
        return uploaded.name

    async def _build_workflow_image_inputs(
        self,
        request: GenerateImageRequest,
    ) -> tuple[comfy_workflow.WorkflowImageInput, ...]:
        slots: list[comfy_workflow.WorkflowImageInput] = []

        for index, raw in enumerate(request.reference_images):
            blob = canonicalize_reference_png_for_comfy(raw, index=index)
            name = await self._upload_png_input(
                blob,
                upload_name=f"ref_{index}.png",
                label="reference",
                index=index,
            )
            slots.append(comfy_workflow.WorkflowImageInput(name, "reference"))

        if request.sketch is not None:
            from image_reference import canonicalize_png_for_comfy

            blob = canonicalize_png_for_comfy(request.sketch, role="sketch")
            name = await self._upload_png_input(
                blob,
                upload_name="sketch.png",
                label="sketch",
            )
            slots.append(comfy_workflow.WorkflowImageInput(name, "sketch"))

        if request.mask is not None:
            from image_reference import canonicalize_png_for_comfy

            blob = canonicalize_png_for_comfy(request.mask, role="mask")
            name = await self._upload_png_input(
                blob,
                upload_name="mask.png",
                label="mask",
            )
            slots.append(comfy_workflow.WorkflowImageInput(name, "mask"))

        return tuple(slots)

    async def generate(
        self,
        request: GenerateImageRequest,
    ) -> ImageGenerationResult:
        validated = self._validate_request(request)
        client = self._client_instance()
        safety = self._safety_validator()

        safety_images: list[SafetyImage] = [
            SafetyImage(mime_type=_guess_mime(blob, index), data=blob)
            for index, blob in enumerate(validated.reference_images)
        ]
        if validated.sketch is not None:
            safety_images.append(
                SafetyImage(mime_type="image/png", data=validated.sketch),
            )
        if validated.mask is not None:
            safety_images.append(SafetyImage(mime_type="image/png", data=validated.mask))

        with track_image_stage("safety"):
            safety_result = await safety.validate(
                validated.prompt,
                safety_images,
            )

            if safety.blocks_on_result(safety_result):
                record_image_generation(
                    "safety",
                    "blocked" if safety_result.status == "rejected" else "error",
                )
                raise SafetyRejectedError(
                    safety_result.reason or "prompt blocked by safety policy",
                    categories=safety_result.categories,
                    status=safety_result.status,
                )

            record_image_generation("safety", "success")

        workflow_image_inputs = await self._build_workflow_image_inputs(validated)

        generated: list[GeneratedImage] = []
        seed_used: int | None = validated.seed
        last_prompt_id: str | None = None
        total_bytes = 0

        allowed_samplers = None
        allowed_schedulers = None

        if self._object_info:
            caps = comfy_workflow.extract_sampler_capabilities(self._object_info)
            sampler_list = caps.get("sampler_name")
            scheduler_list = caps.get("scheduler")

            if isinstance(sampler_list, list) and sampler_list:
                allowed_samplers = frozenset(str(item) for item in sampler_list)

            if isinstance(scheduler_list, list) and scheduler_list:
                allowed_schedulers = frozenset(str(item) for item in scheduler_list)

        logger.info(
            "image generate start image_count=%s reference_images=%s output_root=%s",
            validated.image_count,
            len(validated.reference_images),
            self._config.comfyui_output_root,
        )

        with track_image_stage("comfy_execute"):
            for _ in range(validated.image_count):
                request_id = str(uuid.uuid4())
                params = comfy_workflow.WorkflowBuildParams(
                    request_id=request_id,
                    prompt=validated.prompt,
                    negative_prompt=validated.negative_prompt or "",
                    width=validated.width,
                    height=validated.height,
                    seed=validated.seed,
                    steps=validated.steps,
                    cfg=validated.cfg,
                    sampler_name=validated.sampler,
                    scheduler=validated.scheduler,
                    image_inputs=workflow_image_inputs,
                )
                built = comfy_workflow.build_image_workflow(
                    params,
                    self._config,
                    allowed_samplers=allowed_samplers,
                    allowed_schedulers=allowed_schedulers,
                )
                workflow = built.workflow
                seed_used = built.seed_used

                raw_images = await client.run_workflow(
                    workflow,
                    request_id=request_id,
                    timeout=self._config.generate_timeout,
                )
                if raw_images:
                    last_prompt_id = str(raw_images[0].get("prompt_id") or request_id)

                for item in raw_images:
                    blob = item["data"]
                    total_bytes += len(blob)

                    if total_bytes > self._config.max_output_bytes:
                        record_image_generation("generate", "error")
                        raise OutputTooLargeError(
                            "generated output exceeds server byte limit",
                            max_bytes=self._config.max_output_bytes,
                        )

                    generated.append(
                        GeneratedImage(
                            data=blob,
                            mime_type=str(item.get("mime_type", "image/png")),
                            filename=item.get("filename"),
                        )
                    )

        if not generated:
            record_image_generation("generate", "error")
            logger.warning("image generate finished with zero images after comfy_execute")
            raise InvalidRequestError("ComfyUI returned no images")

        record_image_generation("generate", "success")
        total_bytes = sum(len(img.data) for img in generated)
        logger.info(
            "image generate ok images=%s bytes=%s seed=%s prompt_id=%s",
            len(generated),
            total_bytes,
            seed_used,
            last_prompt_id,
        )

        return ImageGenerationResult(
            images=generated,
            seed=seed_used,
            prompt_id=last_prompt_id,
            timings={},
        )

    async def capabilities(self) -> dict[str, Any]:
        if self._capabilities_cache is not None:
            return self._capabilities_cache

        policy = {
            "backend": "comfyui",
            "max_images": self._config.max_images,
            "max_output_bytes": self._config.max_output_bytes,
            "max_reference_images": self._config.max_reference_images,
            "max_reference_bytes": self._config.max_reference_bytes,
            "inputs": {
                "reference_images": {
                    "supported": True,
                    "max_count": self._config.max_reference_images,
                    "max_bytes": self._config.max_reference_bytes,
                },
                "mask": {
                    "supported": True,
                    "max_count": 1,
                    "max_bytes": self._config.max_reference_bytes,
                    "semantics": "soft_region_guidance",
                },
                "sketch": {
                    "supported": True,
                    "max_count": 1,
                    "max_bytes": self._config.max_reference_bytes,
                    "semantics": "composition_guidance",
                },
            },
            "resolution": {
                "min": self._config.min_resolution,
                "max": self._config.max_resolution,
            },
            "defaults": {
                "resolution": self._config.default_resolution,
                "steps": self._config.default_steps,
                "cfg": self._config.default_cfg,
                "sampler": self._config.default_sampler,
                "scheduler": self._config.default_scheduler,
            },
            "safety_validation_enabled": self._config.safety_enabled,
            "safety_enabled": self._config.safety_enabled,
        }

        samplers: dict[str, Any] = {}

        try:
            object_info = self._object_info

            if object_info is None:
                object_info = await self._client_instance().fetch_object_info()

            samplers = comfy_workflow.extract_sampler_capabilities(object_info)
        except Exception as error:
            logger.debug("capabilities sampler merge skipped: %s", error)

        payload = {
            **policy,
            "samplers": samplers,
        }
        self._capabilities_cache = payload
        return payload


async def create_comfy_backend_if_ready() -> ComfyUIBackend | None:
    """Construct backend and probe; return instance only when probe succeeds."""
    config = load_image_generation_config()

    if not config.enabled:
        return None

    backend = ComfyUIBackend(config)

    if await backend.probe():
        return backend

    return None
