#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIMS = {
    "retrieval_config.py": "from radius_rag.observability.config import *  # noqa: F403\n",
    "retrieval_context.py": "from radius_rag.observability.context import *  # noqa: F403\n",
    "retrieval_candidate.py": "from radius_rag.observability.candidate import *  # noqa: F403\n",
    "retrieval_sanitize.py": "from radius_rag.observability.sanitize import *  # noqa: F403\n",
    "retrieval_snapshot.py": "from radius_rag.observability.snapshot import *  # noqa: F403\n",
    "retrieval_trace.py": "from radius_rag.observability.trace import *  # noqa: F403\n",
    "retrieval_index_info.py": "from radius_rag.observability.index_info import *  # noqa: F403\n",
    "retrieval_event.py": (
        '"""Compatibility shim."""\n'
        "from radius_rag.observability.event import maybe_emit_retrieval_event\n"
        "from radius_rag.observability.sanitize import sanitize_payload\n"
        "__all__ = ['maybe_emit_retrieval_event', 'sanitize_payload']\n"
    ),
    "retrieval_telemetry_metrics.py": "from radius_rag.observability.telemetry.metrics import *  # noqa: F403\n",
    "retrieval_telemetry.py": "from radius_rag.observability.telemetry.writer import *  # noqa: F403\n",
    "retrieval_db.py": "from radius_rag.observability.storage.db import *  # noqa: F403\n",
    "retrieval_judge.py": "from radius_rag.observability.evaluation.judge import *  # noqa: F403\n",
    "retrieval_eval_worker.py": "from radius_rag.observability.evaluation.worker import *  # noqa: F403\n",
    "retrieval_evaluator.py": "from radius_rag.observability.evaluation.evaluator import *  # noqa: F403\n",
    "retrieval_quality_metrics.py": "from radius_rag.observability.evaluation.quality_metrics import *  # noqa: F403\n",
    "retrieval_analytics.py": "from radius_rag.observability.analytics.analytics import *  # noqa: F403\n",
    "retrieval_export.py": "from radius_rag.observability.analytics.export import *  # noqa: F403\n",
    "retrieval_admin.py": "from radius_rag.observability.analytics.admin import *  # noqa: F403\n",
}

HEADER = '"""Backward-compatible shim; see radius_rag.observability."""\n\n'

for name, body in SHIMS.items():
    text = body if body.startswith('"""') else HEADER + body
    (ROOT / name).write_text(text, encoding="utf-8")

print("wrote", len(SHIMS), "shims")
