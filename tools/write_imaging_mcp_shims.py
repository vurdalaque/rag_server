#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGING = [
    "image_analysis",
    "image_concurrency",
    "image_generation",
    "image_generation_config",
    "image_generation_errors",
    "image_reference",
    "image_safety",
    "image_segmentation",
    "image_segmentation_config",
    "image_segmentation_types",
    "image_timings",
    "image_upscale",
    "image_upscale_config",
    "comfy_client",
    "comfy_progress",
    "comfy_segment_workflow",
    "comfy_segmentation",
    "comfy_workflow",
]

MCP = [
    "image_mcp_backends",
    "image_mcp_capabilities",
    "image_mcp_ops",
    "image_mcp_schemas",
    "image_mcp_tools",
    "mcp_test_helpers",
]

HEADER_I = '"""Backward-compatible shim; see radius_rag.imaging."""\n\n'
HEADER_M = '"""Backward-compatible shim; see radius_rag.mcp."""\n\n'

for name in IMAGING:
    (ROOT / f"{name}.py").write_text(
        HEADER_I + f"from radius_rag.imaging.{name} import *  # noqa: F403\n",
        encoding="utf-8",
    )
for name in MCP:
    (ROOT / f"{name}.py").write_text(
        HEADER_M + f"from radius_rag.mcp.{name} import *  # noqa: F403\n",
        encoding="utf-8",
    )

print("wrote", len(IMAGING) + len(MCP), "imaging/mcp shims")
