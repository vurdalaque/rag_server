"""Environment-backed configuration for optional image upscaling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from image_generation_config import ImageGenerationConfig
from rag_metrics import env_bool

_MODULE_DIR = Path(__file__).resolve().parent


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class ImageUpscaleConfig:
    """Server policy and upscale backend connectivity."""

    enabled: bool
    comfy_base_url: str
    comfyui_output_root: Path
    http_upscale_url: str | None

    probe_timeout: float
    upscale_timeout: float
    upload_timeout_seconds: float
    output_read_timeout_seconds: float

    default_scale: float
    max_scale: float
    min_scale: float

    max_input_bytes: int
    max_output_bytes: int
    max_input_dimension: int
    max_output_dimension: int

    allowed_input_mime_types: frozenset[str]

    comfy_upscale_model: str
    comfy_upscale_model_x2: str
    supported_sr_scales: tuple[float, ...]

    @property
    def uses_http_backend(self) -> bool:
        return bool(self.http_upscale_url)


def load_image_upscale_config() -> ImageUpscaleConfig:
    comfy_url = (
        os.getenv("COMFYUI_URL")
        or os.getenv("COMFYUI_BASE_URL")
        or "http://127.0.0.1:8188"
    ).rstrip("/")
    output_root = os.getenv("COMFYUI_OUTPUT_ROOT", "")
    http_url_raw = os.getenv("IMAGE_UPSCALE_URL", "").strip()
    http_url = http_url_raw.rstrip("/") if http_url_raw else None

    mime_raw = os.getenv(
        "IMAGE_UPSCALE_ALLOWED_INPUT_MIME_TYPES",
        os.getenv(
            "IMAGE_GENERATION_ALLOWED_INPUT_MIME_TYPES",
            "image/png,image/jpeg,image/webp",
        ),
    )
    mimes = frozenset(
        part.strip().lower()
        for part in mime_raw.split(",")
        if part.strip()
    )

    upscale_timeout = _env_float(
        "IMAGE_UPSCALE_TIMEOUT",
        _env_float("IMAGE_GENERATION_TIMEOUT", _env_float("COMFYUI_GENERATE_TIMEOUT", 600.0)),
    )

    return ImageUpscaleConfig(
        enabled=env_bool("IMAGE_UPSCALE_ENABLED", False),
        comfy_base_url=comfy_url,
        comfyui_output_root=Path(output_root) if output_root else Path("."),
        http_upscale_url=http_url,
        probe_timeout=_env_float(
            "IMAGE_UPSCALE_PROBE_TIMEOUT",
            _env_float(
                "IMAGE_GENERATION_PROBE_TIMEOUT",
                _env_float("COMFYUI_PROBE_TIMEOUT", 15.0),
            ),
        ),
        upscale_timeout=upscale_timeout,
        upload_timeout_seconds=_env_float(
            "IMAGE_UPSCALE_UPLOAD_TIMEOUT",
            _env_float("IMAGE_GENERATION_UPLOAD_TIMEOUT", 120.0),
        ),
        output_read_timeout_seconds=_env_float(
            "IMAGE_UPSCALE_OUTPUT_READ_TIMEOUT",
            _env_float("IMAGE_GENERATION_OUTPUT_READ_TIMEOUT", 120.0),
        ),
        default_scale=max(1.0, _env_float("IMAGE_UPSCALE_DEFAULT_SCALE", 4.0)),
        max_scale=max(1.0, _env_float("IMAGE_UPSCALE_MAX_SCALE", 4.0)),
        min_scale=max(1.0, _env_float("IMAGE_UPSCALE_MIN_SCALE", 4.0)),
        max_input_bytes=max(
            1,
            _env_int(
                "IMAGE_UPSCALE_MAX_INPUT_BYTES",
                _env_int(
                    "IMAGE_GENERATION_MAX_INPUT_BYTES",
                    _env_int("IMAGE_GENERATION_MAX_REFERENCE_BYTES", 10 * 1024 * 1024),
                ),
            ),
        ),
        max_output_bytes=max(
            1,
            _env_int(
                "IMAGE_UPSCALE_MAX_OUTPUT_BYTES",
                _env_int("IMAGE_GENERATION_MAX_OUTPUT_BYTES", 20 * 1024 * 1024),
            ),
        ),
        max_input_dimension=max(
            1,
            _env_int(
                "IMAGE_UPSCALE_MAX_INPUT_DIMENSION",
                _env_int("IMAGE_GENERATION_MAX_RESOLUTION", 2048),
            ),
        ),
        max_output_dimension=max(
            1,
            _env_int("IMAGE_UPSCALE_MAX_OUTPUT_DIMENSION", 8192),
        ),
        allowed_input_mime_types=mimes,
        comfy_upscale_model=os.getenv(
            "IMAGE_UPSCALE_COMFY_MODEL",
            "RealESRGAN_x4plus.pth",
        ).strip()
        or "RealESRGAN_x4plus.pth",
        comfy_upscale_model_x2=os.getenv(
            "IMAGE_UPSCALE_COMFY_MODEL_X2",
            "RealESRGAN_x2plus.pth",
        ).strip()
        or "RealESRGAN_x2plus.pth",
        supported_sr_scales=(4.0,),
    )


def comfy_generation_config_for_upscale(cfg: ImageUpscaleConfig) -> ImageGenerationConfig:
    """Build ``ImageGenerationConfig`` for ``ComfyUIClient`` connectivity/timeouts."""
    from image_generation_config import load_image_generation_config

    base = load_image_generation_config()
    return ImageGenerationConfig(
        enabled=base.enabled,
        comfy_base_url=cfg.comfy_base_url,
        comfyui_output_root=cfg.comfyui_output_root,
        workflow_template_path=base.workflow_template_path,
        probe_timeout=cfg.probe_timeout,
        generate_timeout=cfg.upscale_timeout,
        upload_timeout_seconds=cfg.upload_timeout_seconds,
        output_read_timeout_seconds=cfg.output_read_timeout_seconds,
        max_images=base.max_images,
        max_output_bytes=cfg.max_output_bytes,
        max_reference_images=base.max_reference_images,
        max_reference_bytes=cfg.max_input_bytes,
        safety_enabled=base.safety_enabled,
        safety_fail_closed=base.safety_fail_closed,
        default_resolution=base.default_resolution,
        min_resolution=base.min_resolution,
        max_resolution=base.max_resolution,
        default_steps=base.default_steps,
        min_steps=base.min_steps,
        max_steps=base.max_steps,
        default_cfg=base.default_cfg,
        min_cfg=base.min_cfg,
        max_cfg=base.max_cfg,
        default_sampler=base.default_sampler,
        default_scheduler=base.default_scheduler,
        default_denoise=base.default_denoise,
        min_denoise=base.min_denoise,
        max_denoise=base.max_denoise,
        allowed_input_mime_types=cfg.allowed_input_mime_types,
    )
