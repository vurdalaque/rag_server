"""Build ComfyUI API workflows for Grounding DINO detect + SAM2 segment (MVP)."""

from __future__ import annotations

import json
from typing import Any

from image_segmentation_config import ImageSegmentationConfig


def _sam2_loader_node(config: ImageSegmentationConfig) -> dict[str, Any]:
    return {
        "inputs": {
            "model": config.sam2_model,
            "segmentor": config.sam2_segmentor,
            "device": config.sam2_device,
            "precision": config.sam2_precision,
        },
        "class_type": config.sam2_loader_class,
    }


def build_grounding_detect_workflow(
    *,
    request_id: str,
    uploaded_filename: str,
    prompt: str,
    config: ImageSegmentationConfig,
) -> dict[str, Any]:
    """Detect highest-confidence bbox; results read from history node ``detect``."""
    return {
        "load": {
            "inputs": {"image": uploaded_filename},
            "class_type": "LoadImage",
        },
        "grounding_model": {
            "inputs": {"model_name": config.grounding_model_name},
            "class_type": config.grounding_loader_class,
        },
        "detect": {
            "inputs": {
                "grounding_model": ["grounding_model", 0],
                "image": ["load", 0],
                "prompt": prompt,
                "threshold": config.grounding_threshold,
            },
            "class_type": config.grounding_detect_class,
        },
    }


def build_sam2_segment_workflow(
    *,
    request_id: str,
    uploaded_filename: str,
    config: ImageSegmentationConfig,
    coordinates_positive: list[dict[str, float]] | None = None,
    coordinates_negative: list[dict[str, float]] | None = None,
    bboxes: list[list[float]] | None = None,
    mask_upload_name: str | None = None,
) -> dict[str, Any]:
    segment_inputs: dict[str, Any] = {
        "sam2_model": ["sam2_model", 0],
        "image": ["load", 0],
        "keep_model_loaded": True,
        "individual_objects": False,
    }
    if coordinates_positive:
        segment_inputs["coordinates_positive"] = json.dumps(coordinates_positive)
    if coordinates_negative:
        segment_inputs["coordinates_negative"] = json.dumps(coordinates_negative)
    if bboxes is not None:
        segment_inputs["bboxes"] = bboxes
    if mask_upload_name is not None:
        segment_inputs["mask"] = ["image_to_mask", 0]

    workflow: dict[str, Any] = {
        "sam2_model": _sam2_loader_node(config),
        "load": {
            "inputs": {"image": uploaded_filename},
            "class_type": "LoadImage",
        },
        "segment": {
            "inputs": segment_inputs,
            "class_type": config.sam2_segment_class,
        },
        "mask_to_image": {
            "inputs": {"mask": ["segment", 0]},
            "class_type": "MaskToImage",
        },
        "save": {
            "inputs": {
                "filename_prefix": f"mcp/segment/{request_id}/mask",
                "images": ["mask_to_image", 0],
            },
            "class_type": "SaveImage",
        },
    }
    if mask_upload_name is not None:
        workflow["load_mask"] = {
            "inputs": {"image": mask_upload_name},
            "class_type": "LoadImage",
        }
        workflow["image_to_mask"] = {
            "inputs": {"image": ["load_mask", 0], "channel": "red"},
            "class_type": "ImageToMask",
        }
    return workflow
