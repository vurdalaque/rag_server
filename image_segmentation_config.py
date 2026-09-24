"""ComfyUI-backed segmentation configuration (SAM2 + Grounding DINO)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from llm_params import env_bool, env_int


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(frozen=True)
class ImageSegmentationConfig:
    enabled: bool
    probe_timeout: float
    segment_timeout: float
    max_bytes: int
    max_dimension: int
    grounding_threshold: float
    grounding_model_name: str
    grounding_loader_class: str
    grounding_detect_class: str
    sam2_model: str
    sam2_segmentor: str
    sam2_device: str
    sam2_precision: str
    sam2_loader_class: str
    sam2_segment_class: str


def load_image_segmentation_config() -> ImageSegmentationConfig:
    return ImageSegmentationConfig(
        enabled=env_bool("IMAGE_SEGMENTATION_ENABLED", True),
        probe_timeout=_env_float("IMAGE_SEGMENTATION_PROBE_TIMEOUT", 15.0),
        segment_timeout=_env_float("IMAGE_SEGMENTATION_TIMEOUT", 600.0),
        max_bytes=max(1, env_int("IMAGE_SEGMENTATION_MAX_BYTES", 10 * 1024 * 1024)),
        max_dimension=max(1, env_int("IMAGE_SEGMENTATION_MAX_DIMENSION", 8192)),
        grounding_threshold=_env_float("IMAGE_GROUNDING_THRESHOLD", 0.3),
        grounding_model_name=os.getenv(
            "IMAGE_GROUNDING_MODEL_NAME",
            "GroundingDINO: SwinT OGC",
        ).strip(),
        grounding_loader_class=os.getenv(
            "IMAGE_GROUNDING_LOADER_CLASS",
            "GroundingModelLoader",
        ).strip(),
        grounding_detect_class=os.getenv(
            "IMAGE_GROUNDING_DETECT_CLASS",
            "GroundingDetector",
        ).strip(),
        sam2_model=os.getenv(
            "IMAGE_SAM2_MODEL",
            "sam2_hiera_small.safetensors",
        ).strip(),
        sam2_segmentor=os.getenv("IMAGE_SAM2_SEGMENTOR", "single_image").strip(),
        sam2_device=os.getenv("IMAGE_SAM2_DEVICE", "cuda").strip(),
        sam2_precision=os.getenv("IMAGE_SAM2_PRECISION", "bf16").strip(),
        sam2_loader_class=os.getenv(
            "IMAGE_SAM2_LOADER_CLASS",
            "DownloadAndLoadSAM2Model",
        ).strip(),
        sam2_segment_class=os.getenv(
            "IMAGE_SAM2_SEGMENT_CLASS",
            "Sam2Segmentation",
        ).strip(),
    )
