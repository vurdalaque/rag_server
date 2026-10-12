"""Register top-level legacy module names (rag_service, retrieval_db, …) without root shims."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

# Keep in sync with docs/refactor_module_map.md (formerly root *.py shims).
LEGACY_MODULE_TARGETS: dict[str, str] = {
    "comfy_client": "radius_rag.imaging.comfy_client",
    "comfy_progress": "radius_rag.imaging.comfy_progress",
    "comfy_segment_workflow": "radius_rag.imaging.comfy_segment_workflow",
    "comfy_segmentation": "radius_rag.imaging.comfy_segmentation",
    "comfy_workflow": "radius_rag.imaging.comfy_workflow",
    "image_analysis": "radius_rag.imaging.image_analysis",
    "image_concurrency": "radius_rag.imaging.image_concurrency",
    "image_generation": "radius_rag.imaging.image_generation",
    "image_generation_config": "radius_rag.imaging.image_generation_config",
    "image_generation_errors": "radius_rag.imaging.image_generation_errors",
    "image_mcp_backends": "radius_rag.mcp.image_mcp_backends",
    "image_mcp_capabilities": "radius_rag.mcp.image_mcp_capabilities",
    "image_mcp_ops": "radius_rag.mcp.image_mcp_ops",
    "image_mcp_schemas": "radius_rag.mcp.image_mcp_schemas",
    "image_mcp_tools": "radius_rag.mcp.image_mcp_tools",
    "image_reference": "radius_rag.imaging.image_reference",
    "image_safety": "radius_rag.imaging.image_safety",
    "image_segmentation": "radius_rag.imaging.image_segmentation",
    "image_segmentation_config": "radius_rag.imaging.image_segmentation_config",
    "image_segmentation_types": "radius_rag.imaging.image_segmentation_types",
    "image_timings": "radius_rag.imaging.image_timings",
    "image_upscale": "radius_rag.imaging.image_upscale",
    "image_upscale_config": "radius_rag.imaging.image_upscale_config",
    "llm_client": "radius_rag.llm.client",
    "llm_params": "radius_rag.llm.params",
    "rag_bundle": "radius_rag.rag.bundle",
    "rag_metrics": "radius_rag.rag.metrics",
    "rag_proxy": "radius_rag.rag.proxy",
    "rag_search": "radius_rag.rag.search",
    "rag_service": "radius_rag.rag.service",
    "retrieval_admin": "radius_rag.observability.analytics.admin",
    "retrieval_analytics": "radius_rag.observability.analytics.analytics",
    "retrieval_candidate": "radius_rag.observability.candidate",
    "retrieval_config": "radius_rag.observability.config",
    "retrieval_context": "radius_rag.observability.context",
    "retrieval_db": "radius_rag.observability.storage.db",
    "retrieval_eval_worker": "radius_rag.observability.evaluation.worker",
    "retrieval_evaluator": "radius_rag.observability.evaluation.evaluator",
    "retrieval_event": "radius_rag.observability.event",
    "retrieval_export": "radius_rag.observability.analytics.export",
    "retrieval_index_info": "radius_rag.observability.index_info",
    "retrieval_judge": "radius_rag.observability.evaluation.judge",
    "retrieval_quality_metrics": "radius_rag.observability.evaluation.quality_metrics",
    "retrieval_sanitize": "radius_rag.observability.sanitize",
    "retrieval_snapshot": "radius_rag.observability.snapshot",
    "retrieval_telemetry": "radius_rag.observability.telemetry.writer",
    "retrieval_telemetry_metrics": "radius_rag.observability.telemetry.metrics",
    "retrieval_trace": "radius_rag.observability.trace",
}

_installed = False
_finder: importlib.abc.MetaPathFinder | None = None


class LegacyModuleLoader(importlib.abc.Loader):
    def __init__(self, legacy_name: str, target_name: str) -> None:
        self._legacy_name = legacy_name
        self._target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        target = importlib.import_module(self._target_name)
        # Alias legacy top-level name to the packaged module (single object).
        sys.modules[self._legacy_name] = target
        module.__dict__.update(target.__dict__)


class LegacyModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        target_name = LEGACY_MODULE_TARGETS.get(fullname)
        if target_name is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            LegacyModuleLoader(fullname, target_name),
            is_package=False,
        )


def install_legacy_imports() -> None:
    global _installed, _finder
    if _installed:
        return
    _finder = LegacyModuleFinder()
    sys.meta_path.insert(0, _finder)
    _installed = True
