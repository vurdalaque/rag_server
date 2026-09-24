"""Image upscale backend facade (ComfyUI or optional HTTP service)."""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

from comfy_client import ComfyUIClient
from image_generation_errors import (
    BackendUnavailableError,
    ComfyUIUnavailableError,
    ImageGenerationError,
    InputTooLargeError,
    InvalidRequestError,
    OutputTooLargeError,
    UnsupportedParameterError,
    UpscaleFailedError,
)
from comfy_workflow import _combo_options
from image_concurrency import comfy_gpu_slot
from image_timings import elapsed_ms, merge_timings_ms
from image_upscale_config import (
    ImageUpscaleConfig,
    comfy_generation_config_for_upscale,
    load_image_upscale_config,
)

logger = logging.getLogger(__name__)

_SR_NODE_CLASSES = frozenset(
    {"UpscaleModelLoader", "ImageUpscaleWithModel", "LoadImage", "SaveImage"},
)

ScaleMode = Literal["factor", "target"]


@dataclass(frozen=True)
class UpscaleRequest:
    image: bytes
    scale: float | None = None
    target_width: int | None = None
    target_height: int | None = None


@dataclass(frozen=True)
class UpscaleResult:
    image: bytes
    width: int
    height: int
    timings: dict[str, float]


@dataclass(frozen=True)
class _ResolvedUpscale:
    mode: ScaleMode
    scale: float | None
    target_width: int | None
    target_height: int | None
    source_width: int
    source_height: int


@runtime_checkable
class ImageUpscaler(Protocol):
    async def probe(self) -> bool:
        """Check upstream readiness without running inference."""

    async def upscale(self, request: UpscaleRequest) -> UpscaleResult:
        """Validate input and run upscaling."""

    async def capabilities(self) -> dict[str, Any]:
        """Return model-agnostic limits for discovery."""


def _guess_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image/webp"
    guessed, _ = mimetypes.guess_type("input.bin")
    return guessed or "application/octet-stream"


def _decode_image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as error:
        raise InvalidRequestError(
            "server cannot validate images (Pillow not installed)",
        ) from error

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except OSError as error:
        raise InvalidRequestError(
            "image is corrupt or truncated",
            reason=str(error),
            size_bytes=len(data),
        ) from error
    except Exception as error:
        raise InvalidRequestError(
            "image could not be decoded",
            reason=str(error),
            size_bytes=len(data),
        ) from error

    if width < 1 or height < 1:
        raise InvalidRequestError("image has invalid dimensions", width=width, height=height)
    return width, height


def _canonicalize_png(data: bytes) -> bytes:
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            converted = image.convert("RGBA")
        elif image.mode != "RGB":
            converted = image.convert("RGB")
        else:
            converted = image.copy()
        buffer = BytesIO()
        converted.save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue()


def _resolve_upscale_params(
    request: UpscaleRequest,
    config: ImageUpscaleConfig,
) -> _ResolvedUpscale:
    has_scale = request.scale is not None
    has_target = request.target_width is not None or request.target_height is not None

    if has_target:
        raise UnsupportedParameterError(
            "target_width and target_height are not supported in this server version",
        )

    source_width, source_height = _decode_image_dimensions(request.image)

    if len(request.image) > config.max_input_bytes:
        raise InputTooLargeError(
            "input image exceeds server byte limit",
            max_bytes=config.max_input_bytes,
            size_bytes=len(request.image),
        )

    if source_width > config.max_input_dimension or source_height > config.max_input_dimension:
        raise InputTooLargeError(
            "input image dimensions exceed server limit",
            max_dimension=config.max_input_dimension,
            width=source_width,
            height=source_height,
        )

    if not has_scale and not has_target:
        scale = config.default_scale
        return _ResolvedUpscale(
            mode="factor",
            scale=scale,
            target_width=None,
            target_height=None,
            source_width=source_width,
            source_height=source_height,
        )

    if has_scale:
        scale = float(request.scale)  # type: ignore[arg-type]
        if scale < config.min_scale or scale > config.max_scale:
            raise UnsupportedParameterError(
                "scale is outside supported range",
                scale=scale,
                min_scale=config.min_scale,
                max_scale=config.max_scale,
            )
        out_w = max(1, int(round(source_width * scale)))
        out_h = max(1, int(round(source_height * scale)))
        _assert_output_dimensions(out_w, out_h, config)
        return _ResolvedUpscale(
            mode="factor",
            scale=scale,
            target_width=None,
            target_height=None,
            source_width=source_width,
            source_height=source_height,
        )

    target_width = request.target_width
    target_height = request.target_height
    if target_width is not None and target_width < 1:
        raise InvalidRequestError("target_width must be positive")
    if target_height is not None and target_height < 1:
        raise InvalidRequestError("target_height must be positive")
    if target_width is None and target_height is None:
        raise InvalidRequestError("target_width or target_height is required")

    if target_width is None:
        target_width = max(1, int(round(source_width * (target_height / source_height))))
    elif target_height is None:
        target_height = max(1, int(round(source_height * (target_width / source_width))))

    _assert_output_dimensions(target_width, target_height, config)
    if target_width <= source_width and target_height <= source_height:
        raise UnsupportedParameterError(
            "target dimensions must increase resolution relative to the source image",
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )

    return _ResolvedUpscale(
        mode="target",
        scale=None,
        target_width=target_width,
        target_height=target_height,
        source_width=source_width,
        source_height=source_height,
    )


def _assert_output_dimensions(width: int, height: int, config: ImageUpscaleConfig) -> None:
    if width > config.max_output_dimension or height > config.max_output_dimension:
        raise OutputTooLargeError(
            "requested output dimensions exceed server limit",
            max_dimension=config.max_output_dimension,
            width=width,
            height=height,
        )


def _assert_output_bytes(data: bytes, config: ImageUpscaleConfig) -> None:
    if len(data) > config.max_output_bytes:
        raise OutputTooLargeError(
            "upscaled output exceeds server byte limit",
            max_bytes=config.max_output_bytes,
            size_bytes=len(data),
        )


def _list_upscale_models(object_info: dict[str, Any]) -> list[str]:
    loader = object_info.get("UpscaleModelLoader")
    if not isinstance(loader, dict):
        return []
    required = loader.get("input", {}).get("required", {})
    return _combo_options(required.get("model_name"))


def _comfy_sr_supported(object_info: dict[str, Any], config: ImageUpscaleConfig) -> bool:
    if not _SR_NODE_CLASSES <= set(object_info.keys()):
        return False
    models = _list_upscale_models(object_info)
    if not models:
        return False
    needed = {config.comfy_upscale_model, config.comfy_upscale_model_x2}
    return bool(needed & set(models))


def _effective_scale(resolved: _ResolvedUpscale) -> float:
    if resolved.mode == "factor" and resolved.scale is not None:
        return float(resolved.scale)
    if resolved.target_width is None or resolved.target_height is None:
        raise UpscaleFailedError("missing target dimensions for upscale")
    return max(
        resolved.target_width / resolved.source_width,
        resolved.target_height / resolved.source_height,
    )


def _pick_sr_model_name(
    resolved: _ResolvedUpscale,
    config: ImageUpscaleConfig,
    available_models: list[str],
) -> tuple[str, float]:
    scale = _effective_scale(resolved)
    tolerance = 0.06
    for supported in config.supported_sr_scales:
        if abs(scale - supported) <= tolerance:
            if supported <= 2.0 + tolerance:
                name = config.comfy_upscale_model_x2
            else:
                name = config.comfy_upscale_model
            if name in available_models:
                return name, supported
    raise UnsupportedParameterError(
        "requested upscale factor is not supported by configured SR models",
        scale=scale,
        supported_scales=list(config.supported_sr_scales),
    )


def build_comfy_upscale_workflow(
    *,
    request_id: str,
    uploaded_filename: str,
    model_name: str,
) -> dict[str, Any]:
    load_node = "1"
    loader_node = "2"
    upscale_node = "3"
    save_node = "4"

    return {
        load_node: {
            "inputs": {"image": uploaded_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "LoadImage (upscale input)"},
        },
        loader_node: {
            "inputs": {"model_name": model_name},
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "UpscaleModelLoader"},
        },
        upscale_node: {
            "inputs": {
                "upscale_model": [loader_node, 0],
                "image": [load_node, 0],
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "ImageUpscaleWithModel"},
        },
        save_node: {
            "inputs": {
                "filename_prefix": f"mcp/upscale/{request_id}/result",
                "images": [upscale_node, 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "SaveImage (upscale output)"},
        },
    }


class ComfyUpscaleBackend:
    """Upscale via a minimal ComfyUI LoadImage -> scale -> SaveImage workflow."""

    def __init__(
        self,
        config: ImageUpscaleConfig | None = None,
        *,
        client: ComfyUIClient | None = None,
    ) -> None:
        self._config = config or load_image_upscale_config()
        self._comfy_config = comfy_generation_config_for_upscale(self._config)
        self._client = client
        self._object_info: dict[str, Any] | None = None
        self._available_models: list[str] = []

    def _client_instance(self) -> ComfyUIClient:
        if self._client is not None:
            return self._client
        return ComfyUIClient(self._comfy_config)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def probe(self) -> bool:
        if not self._config.enabled:
            return False
        if self._config.uses_http_backend and not self._config.comfy_base_url:
            return False

        try:
            client = self._client_instance()
            object_info = await client.fetch_object_info(timeout=self._config.probe_timeout)
            if "LoadImage" not in object_info or "SaveImage" not in object_info:
                logger.warning("ComfyUI upscale probe missing LoadImage or SaveImage nodes")
                return False
            if not _comfy_sr_supported(object_info, self._config):
                logger.warning(
                    "ComfyUI upscale probe: Real-ESRGAN SR nodes/models not available",
                )
                return False
            self._object_info = object_info
            self._available_models = _list_upscale_models(object_info)
        except ImageGenerationError as error:
            logger.warning("Image upscale ComfyUI probe failed: %s", error.message)
            return False
        except Exception as error:
            logger.warning("Image upscale ComfyUI probe failed: %s", error)
            return False

        logger.info(
            "image upscale probe ok sr_models=%s output_root=%s",
            self._available_models,
            self._config.comfyui_output_root,
        )
        return True

    def _validate_input_mime(self, data: bytes) -> None:
        mime = _guess_mime(data)
        if mime not in self._config.allowed_input_mime_types:
            raise InvalidRequestError(
                "unsupported input image type",
                mime_type=mime,
                allowed=sorted(self._config.allowed_input_mime_types),
            )

    async def upscale(self, request: UpscaleRequest) -> UpscaleResult:
        timings: dict[str, float] = {}
        started = time.monotonic()

        self._validate_input_mime(request.image)
        t0 = time.monotonic()
        resolved = _resolve_upscale_params(request, self._config)
        timings["validate"] = time.monotonic() - t0

        png_bytes = _canonicalize_png(request.image)
        request_id = str(uuid.uuid4())
        client = self._client_instance()

        if not self._available_models:
            if self._object_info is None:
                self._object_info = await client.fetch_object_info()
                self._available_models = _list_upscale_models(self._object_info)
        if not self._available_models:
            raise UpscaleFailedError("ComfyUI SR upscale models are not available")

        model_name, _ = _pick_sr_model_name(
            resolved,
            self._config,
            self._available_models,
        )

        t0 = time.monotonic()
        async with comfy_gpu_slot():
            try:
                uploaded = await client.upload_image(
                    f"upscale_{request_id}.png",
                    png_bytes,
                    "image/png",
                )
            except ComfyUIUnavailableError as error:
                raise UpscaleFailedError(
                    "failed to upload image to ComfyUI",
                    reason=error.message,
                ) from error
            timings["upload_ms"] = elapsed_ms(t0)

            workflow = build_comfy_upscale_workflow(
                request_id=request_id,
                uploaded_filename=upload.name,
                model_name=model_name,
            )

            t_exec = time.monotonic()
            try:
                raw_outputs = await client.run_workflow(
                    workflow,
                    request_id=request_id,
                    timeout=self._config.upscale_timeout,
                )
            except ImageGenerationError as error:
                raise UpscaleFailedError(
                    "ComfyUI upscale failed",
                    reason=error.message,
                    code=error.code,
                ) from error
            timings["execute_ms"] = elapsed_ms(t_exec)

        if not raw_outputs:
            raise UpscaleFailedError("ComfyUI upscale returned no images")

        blob = raw_outputs[0]["data"]
        _assert_output_bytes(blob, self._config)
        out_width, out_height = _decode_image_dimensions(blob)
        timings = merge_timings_ms(total=elapsed_ms(started), **timings)

        return UpscaleResult(
            image=blob,
            width=out_width,
            height=out_height,
            timings=timings,
        )

    async def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "comfyui",
            "method": "super_resolution",
            "models": list(self._available_models),
            "supported_scale_factors": list(self._config.supported_sr_scales),
            "max_scale": self._config.max_scale,
            "min_scale": self._config.min_scale,
            "default_scale": self._config.default_scale,
            "max_input_bytes": self._config.max_input_bytes,
            "max_output_bytes": self._config.max_output_bytes,
            "max_input_dimension": self._config.max_input_dimension,
            "max_output_dimension": self._config.max_output_dimension,
            "supports_target_dimensions": False,
            "supports_scale_factor": True,
            "scales": list(self._config.supported_sr_scales),
        }


async def create_upscale_backend_if_ready() -> ImageUpscaler | None:
    """Construct ComfyUI Real-ESRGAN upscale backend when enabled and probe succeeds."""
    config = load_image_upscale_config()
    if not config.enabled:
        return None

    backend = ComfyUpscaleBackend(config)
    if await backend.probe():
        return backend
    await backend.aclose()
    return None
