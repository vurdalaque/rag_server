"""Environment-backed configuration for optional image generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from rag_metrics import env_bool

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_TEMPLATE = _MODULE_DIR / "data" / "comfy_image_workflow.template.json"


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
class ImageGenerationConfig:
    """Server policy and ComfyUI connectivity."""

    enabled: bool
    comfy_base_url: str
    comfyui_output_root: Path
    workflow_template_path: Path

    probe_timeout: float
    generate_timeout: float
    upload_timeout_seconds: float
    output_read_timeout_seconds: float

    max_images: int
    max_output_bytes: int
    max_reference_images: int
    max_reference_bytes: int

    safety_enabled: bool
    safety_fail_closed: bool

    default_resolution: int
    min_resolution: int
    max_resolution: int
    default_steps: int
    min_steps: int
    max_steps: int
    default_cfg: float
    min_cfg: float
    max_cfg: float
    default_sampler: str
    default_scheduler: str
    default_denoise: float
    min_denoise: float
    max_denoise: float

    allowed_input_mime_types: frozenset[str]

    @property
    def comfyui_url(self) -> str:
        return self.comfy_base_url

    @property
    def timeout_seconds(self) -> float:
        return self.generate_timeout

    @property
    def probe_timeout_seconds(self) -> float:
        return self.probe_timeout

    @property
    def ws_timeout_seconds(self) -> float:
        return self.generate_timeout

    @property
    def max_input_images(self) -> int:
        return self.max_reference_images

    @property
    def max_input_bytes(self) -> int:
        return self.max_reference_bytes

    def comfy_ws_url(self, client_id: str) -> str:
        parsed = urlparse(self.comfy_base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        host = parsed.netloc or parsed.path
        return f"{scheme}://{host}/ws?clientId={client_id}"


def load_image_generation_config() -> ImageGenerationConfig:
    template_raw = os.getenv(
        "COMFYUI_WORKFLOW_TEMPLATE",
        str(_DEFAULT_TEMPLATE),
    )
    comfy_url = (
        os.getenv("COMFYUI_URL")
        or os.getenv("COMFYUI_BASE_URL")
        or "http://127.0.0.1:8188"
    ).rstrip("/")
    output_root = os.getenv("COMFYUI_OUTPUT_ROOT", "")
    mime_raw = os.getenv(
        "IMAGE_GENERATION_ALLOWED_INPUT_MIME_TYPES",
        "image/png,image/jpeg,image/webp",
    )
    mimes = frozenset(
        part.strip().lower()
        for part in mime_raw.split(",")
        if part.strip()
    )
    generate_timeout = _env_float(
        "IMAGE_GENERATION_TIMEOUT",
        _env_float("COMFYUI_GENERATE_TIMEOUT", 600.0),
    )
    max_reference = max(
        0,
        _env_int(
            "IMAGE_GENERATION_MAX_INPUT_IMAGES",
            _env_int("IMAGE_GENERATION_MAX_REFERENCE_IMAGES", 10),
        ),
    )
    max_reference_bytes = max(
        1,
        _env_int(
            "IMAGE_GENERATION_MAX_INPUT_BYTES",
            _env_int(
                "IMAGE_GENERATION_MAX_REFERENCE_BYTES",
                10 * 1024 * 1024,
            ),
        ),
    )

    return ImageGenerationConfig(
        enabled=env_bool("IMAGE_GENERATION_ENABLED", False),
        comfy_base_url=comfy_url,
        comfyui_output_root=Path(output_root) if output_root else Path("."),
        workflow_template_path=Path(template_raw),
        probe_timeout=_env_float(
            "IMAGE_GENERATION_PROBE_TIMEOUT",
            _env_float("COMFYUI_PROBE_TIMEOUT", 15.0),
        ),
        generate_timeout=generate_timeout,
        upload_timeout_seconds=_env_float("IMAGE_GENERATION_UPLOAD_TIMEOUT", 120.0),
        output_read_timeout_seconds=_env_float(
            "IMAGE_GENERATION_OUTPUT_READ_TIMEOUT",
            120.0,
        ),
        max_images=max(1, _env_int("IMAGE_GENERATION_MAX_IMAGES", 4)),
        max_output_bytes=max(
            1,
            _env_int(
                "IMAGE_GENERATION_MAX_OUTPUT_BYTES",
                20 * 1024 * 1024,
            ),
        ),
        max_reference_images=max_reference,
        max_reference_bytes=max_reference_bytes,
        safety_enabled=env_bool("IMAGE_GENERATION_SAFETY_ENABLED", False),
        safety_fail_closed=env_bool("IMAGE_GENERATION_SAFETY_FAIL_CLOSED", True),
        default_resolution=max(
            64,
            _env_int("IMAGE_GENERATION_DEFAULT_RESOLUTION", 1024),
        ),
        min_resolution=_env_int("IMAGE_GENERATION_MIN_RESOLUTION", 256),
        max_resolution=_env_int("IMAGE_GENERATION_MAX_RESOLUTION", 2048),
        default_steps=max(1, _env_int("IMAGE_GENERATION_DEFAULT_STEPS", 25)),
        min_steps=_env_int("IMAGE_GENERATION_MIN_STEPS", 1),
        max_steps=_env_int("IMAGE_GENERATION_MAX_STEPS", 100),
        default_cfg=_env_float("IMAGE_GENERATION_DEFAULT_CFG", 1.0),
        min_cfg=_env_float("IMAGE_GENERATION_MIN_CFG", 0.0),
        max_cfg=_env_float("IMAGE_GENERATION_MAX_CFG", 30.0),
        default_sampler=os.getenv("IMAGE_GENERATION_DEFAULT_SAMPLER", "euler"),
        default_scheduler=os.getenv("IMAGE_GENERATION_DEFAULT_SCHEDULER", "simple"),
        default_denoise=_env_float("IMAGE_GENERATION_DEFAULT_DENOISE", 1.0),
        min_denoise=_env_float("IMAGE_GENERATION_MIN_DENOISE", 0.0),
        max_denoise=_env_float("IMAGE_GENERATION_MAX_DENOISE", 1.0),
        allowed_input_mime_types=mimes,
    )
