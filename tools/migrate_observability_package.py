#!/usr/bin/env python3
"""One-off: copy retrieval_* modules into src/radius_rag/observability/."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "radius_rag" / "observability"

MAP = {
    "retrieval_config": "radius_rag.observability.config",
    "retrieval_context": "radius_rag.observability.context",
    "retrieval_candidate": "radius_rag.observability.candidate",
    "retrieval_sanitize": "radius_rag.observability.sanitize",
    "retrieval_snapshot": "radius_rag.observability.snapshot",
    "retrieval_trace": "radius_rag.observability.trace",
    "retrieval_index_info": "radius_rag.observability.index_info",
    "retrieval_event": "radius_rag.observability.event",
    "retrieval_telemetry_metrics": "radius_rag.observability.telemetry.metrics",
    "retrieval_telemetry": "radius_rag.observability.telemetry.writer",
    "retrieval_db": "radius_rag.observability.storage.db",
    "retrieval_judge": "radius_rag.observability.evaluation.judge",
    "retrieval_eval_worker": "radius_rag.observability.evaluation.worker",
    "retrieval_evaluator": "radius_rag.observability.evaluation.evaluator",
    "retrieval_quality_metrics": "radius_rag.observability.evaluation.quality_metrics",
    "retrieval_analytics": "radius_rag.observability.analytics.analytics",
    "retrieval_export": "radius_rag.observability.analytics.export",
    "retrieval_admin": "radius_rag.observability.analytics.admin",
}

FILES = {
    "retrieval_config.py": SRC / "config.py",
    "retrieval_context.py": SRC / "context.py",
    "retrieval_candidate.py": SRC / "candidate.py",
    "retrieval_sanitize.py": SRC / "sanitize.py",
    "retrieval_snapshot.py": SRC / "snapshot.py",
    "retrieval_trace.py": SRC / "trace.py",
    "retrieval_index_info.py": SRC / "index_info.py",
    "retrieval_event.py": SRC / "event.py",
    "retrieval_telemetry_metrics.py": SRC / "telemetry" / "metrics.py",
    "retrieval_telemetry.py": SRC / "telemetry" / "writer.py",
    "retrieval_db.py": SRC / "storage" / "db.py",
    "retrieval_judge.py": SRC / "evaluation" / "judge.py",
    "retrieval_eval_worker.py": SRC / "evaluation" / "worker.py",
    "retrieval_evaluator.py": SRC / "evaluation" / "evaluator.py",
    "retrieval_quality_metrics.py": SRC / "evaluation" / "quality_metrics.py",
    "retrieval_analytics.py": SRC / "analytics" / "analytics.py",
    "retrieval_export.py": SRC / "analytics" / "export.py",
    "retrieval_admin.py": SRC / "analytics" / "admin.py",
}

PAT = re.compile(r"\b(from|import)\s+(retrieval_[a-z_]+)")


def rewrite(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = MAP.get(mod)
        if not new:
            return match.group(0)
        if match.group(1) == "from":
            return f"from {new}"
        return f"import {new}"

    return PAT.sub(repl, text)


def main() -> None:
    for sub in ("telemetry", "storage", "evaluation", "analytics"):
        (SRC / sub).mkdir(parents=True, exist_ok=True)
        init = SRC / sub / "__init__.py"
        if not init.exists():
            init.write_text('"""Observability subpackage."""\n', encoding="utf-8")

    for src_name, dest in FILES.items():
        text = (ROOT / src_name).read_text(encoding="utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rewrite(text), encoding="utf-8")

    # Fix migrations path in packaged config
    config_path = SRC / "config.py"
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "_MODULE_DIR = Path(__file__).resolve().parent\nMIGRATIONS_DIR = _MODULE_DIR / \"retrieval_migrations\"",
        "_RAG_SERVER_ROOT = Path(__file__).resolve().parents[3]\nMIGRATIONS_DIR = Path(\n    os.getenv(\"RETRIEVAL_MIGRATIONS_DIR\", str(_RAG_SERVER_ROOT / \"retrieval_migrations\"))\n)",
    )
    config_path.write_text(config_text, encoding="utf-8")
    print(f"wrote {len(FILES)} modules under {SRC}")


if __name__ == "__main__":
    main()
