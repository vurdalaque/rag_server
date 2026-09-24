"""Build combined image capability payloads for MCP discovery."""

from __future__ import annotations

from typing import Any

from image_analysis import load_image_analysis_config
from image_mcp_backends import ImageMcpBackends
from image_segmentation import load_image_segmentation_config
from image_upscale_config import load_image_upscale_config


def _empty_generation_payload() -> dict[str, Any]:
    return {
        "backend": "",
        "max_images": 0,
        "max_output_bytes": 0,
        "max_reference_images": 0,
        "max_reference_bytes": 0,
        "inputs": {
            "reference_images": {
                "supported": False,
                "max_count": 0,
                "max_bytes": 0,
            },
            "mask": {
                "supported": False,
                "max_count": 0,
                "max_bytes": 0,
                "semantics": "soft_region_guidance",
            },
            "sketch": {
                "supported": False,
                "max_count": 0,
                "max_bytes": 0,
                "semantics": "composition_guidance",
            },
        },
        "resolution": {"min": 0, "max": 0},
        "defaults": {
            "resolution": 0,
            "steps": 0,
            "cfg": 0.0,
            "sampler": "",
            "scheduler": "",
        },
        "safety_validation_enabled": False,
        "safety_enabled": False,
        "samplers": {},
    }


async def build_platform_capabilities_payload(
    backends: ImageMcpBackends,
) -> dict[str, Any]:
    if backends.generation is not None:
        generation = await backends.generation.capabilities()
    else:
        generation = _empty_generation_payload()

    analysis_cfg = load_image_analysis_config()
    segment_cfg = load_image_segmentation_config()
    upscale_cfg = load_image_upscale_config()

    analysis = {
        "supported": backends.analyzer is not None and analysis_cfg.enabled,
        "max_images": analysis_cfg.max_images,
        "max_bytes": analysis_cfg.max_bytes,
    }

    segmentation: dict[str, Any] = {
        "supported": backends.segmenter is not None and segment_cfg.enabled,
        "max_bytes": segment_cfg.max_bytes,
        "max_dimension": segment_cfg.max_dimension,
        "mask_semantics": "selected_white_1_not_selected_black_0",
        "modes": {
            "text": False,
            "points": False,
            "box": False,
            "mask_refinement": False,
        },
    }
    if backends.segmenter is not None:
        try:
            caps = await backends.segmenter.capabilities()
            segmentation.update(
                {
                    "backend": caps.get("backend"),
                    "model": caps.get("model"),
                    "mask_semantics": caps.get(
                        "mask_semantics",
                        segmentation["mask_semantics"],
                    ),
                    "modes": caps.get("modes", segmentation["modes"]),
                },
            )
        except Exception:
            pass

    upscale: dict[str, Any] = {
        "supported": backends.upscaler is not None and upscale_cfg.enabled,
        "max_input_bytes": upscale_cfg.max_input_bytes,
        "max_output_bytes": upscale_cfg.max_output_bytes,
        "max_input_dimension": upscale_cfg.max_input_dimension,
        "max_output_dimension": upscale_cfg.max_output_dimension,
        "supports_scale_factor": True,
        "supports_target_dimensions": True,
    }
    if backends.upscaler is not None:
        try:
            upscale.update(await backends.upscaler.capabilities())
        except Exception:
            pass

    segment_mvp = {
        "supported": segmentation["supported"],
        "text": segmentation["modes"].get("text", False),
        "points": segmentation["modes"].get("points", False),
        "box": segmentation["modes"].get("box", False),
        "refinement": segmentation["modes"].get("mask_refinement", False),
    }
    upscale_mvp = {
        "supported": upscale.get("supported", False),
        "scales": upscale.get("scales") or upscale.get("supported_scale_factors") or [],
    }

    return {
        **generation,
        "analysis": analysis,
        "segmentation": segmentation,
        "upscale": upscale,
        "generate": {"supported": backends.generation is not None},
        "analyze": {"supported": analysis["supported"]},
        "segment": segment_mvp,
    }
