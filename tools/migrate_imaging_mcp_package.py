#!/usr/bin/env python3
"""Copy image_*, comfy_*, image_mcp_* into src/radius_rag/imaging and mcp."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGING = ROOT / "src" / "radius_rag" / "imaging"
MCP = ROOT / "src" / "radius_rag" / "mcp"

IMAGING_MODULES = [
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

MCP_MODULES = [
    "image_mcp_backends",
    "image_mcp_capabilities",
    "image_mcp_ops",
    "image_mcp_schemas",
    "image_mcp_tools",
]

CROSS = {
    "rag_metrics": "radius_rag.rag.metrics",
    "rag_service": "radius_rag.rag.service",
    "rag_bundle": "radius_rag.rag.bundle",
    "llm_params": "radius_rag.llm.params",
    "llm_client": "radius_rag.llm.client",
}

LOCAL = {}
for name in IMAGING_MODULES:
    LOCAL[name] = f"radius_rag.imaging.{name}"
for name in MCP_MODULES:
    LOCAL[name] = f"radius_rag.mcp.{name}"

ALL_LOCAL = set(LOCAL.keys())
PAT = re.compile(r"\b(from|import)\s+(" + "|".join(sorted(ALL_LOCAL, key=len, reverse=True)) + r")\b")
CROSS_PAT = re.compile(
    r"\b(from|import)\s+(" + "|".join(sorted(CROSS.keys(), key=len, reverse=True)) + r")\b"
)


def rewrite(text: str) -> str:
    def repl_local(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = LOCAL.get(mod)
        if not new:
            return match.group(0)
        return f"{match.group(1)} {new}"

    def repl_cross(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = CROSS.get(mod)
        if not new:
            return match.group(0)
        return f"{match.group(1)} {new}"

    text = PAT.sub(repl_local, text)
    return CROSS_PAT.sub(repl_cross, text)


def main() -> None:
    IMAGING.mkdir(parents=True, exist_ok=True)
    MCP.mkdir(parents=True, exist_ok=True)
    for init, blurb in (
        (IMAGING / "__init__.py", '"""Image generation, ComfyUI, segmentation, upscale."""\n'),
        (MCP / "__init__.py", '"""MCP tool wiring for image capabilities."""\n'),
    ):
        if not init.exists():
            init.write_text(blurb, encoding="utf-8")

    for name in IMAGING_MODULES:
        src = ROOT / f"{name}.py"
        dest = IMAGING / f"{name}.py"
        text = src.read_text(encoding="utf-8")
        if "radius_rag.imaging" in text and len(text) < 400:
            raise SystemExit(f"{name}.py looks like a shim")
        dest.write_text(rewrite(text), encoding="utf-8")

    for name in MCP_MODULES:
        src = ROOT / f"{name}.py"
        dest = MCP / f"{name}.py"
        text = src.read_text(encoding="utf-8")
        if "radius_rag.mcp" in text and len(text) < 400:
            raise SystemExit(f"{name}.py looks like a shim")
        dest.write_text(rewrite(text), encoding="utf-8")

    print(f"wrote {len(IMAGING_MODULES)} imaging + {len(MCP_MODULES)} mcp modules")


if __name__ == "__main__":
    main()
