"""Shared helpers for image/MCP tests."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


def sample_capabilities_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": "comfyui",
        "max_images": 4,
        "max_output_bytes": 20971520,
        "max_reference_images": 10,
        "max_reference_bytes": 10485760,
        "inputs": {
            "reference_images": {
                "supported": True,
                "max_count": 10,
                "max_bytes": 10485760,
            },
            "mask": {
                "supported": True,
                "max_count": 1,
                "max_bytes": 10485760,
                "semantics": "soft_region_guidance",
            },
            "sketch": {
                "supported": True,
                "max_count": 1,
                "max_bytes": 10485760,
                "semantics": "composition_guidance",
            },
        },
        "resolution": {"min": 256, "max": 2048},
        "defaults": {
            "resolution": 1024,
            "steps": 25,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
        },
        "safety_validation_enabled": False,
        "safety_enabled": False,
        "samplers": {"sampler_name": ["euler"], "scheduler": ["simple"]},
        "analysis": {"supported": True, "max_images": 10, "max_bytes": 10485760},
        "segmentation": {
            "supported": True,
            "max_bytes": 10485760,
            "max_dimension": 8192,
            "mask_semantics": "selected_white_1_not_selected_black_0",
            "modes": {
                "text": True,
                "points": True,
                "box": True,
                "mask_refinement": True,
            },
        },
        "upscale": {
            "supported": True,
            "max_input_bytes": 10485760,
            "max_output_bytes": 20971520,
            "max_input_dimension": 4096,
            "max_output_dimension": 8192,
            "supports_scale_factor": True,
            "supports_target_dimensions": True,
            "max_scale": 4.0,
            "default_scale": 2.0,
        },
    }
    payload.update(overrides)
    return payload


def png_b64(
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (1, 2, 3),
) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def truncated_png_b64() -> str:
    data = base64.b64decode(png_b64())
    return base64.b64encode(data[:-40]).decode("ascii")
