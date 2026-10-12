#!/usr/bin/env python3
"""Copy rag_* modules into src/radius_rag/rag/ and rewrite internal imports."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "radius_rag" / "rag"

MAP = {
    "rag_metrics": "radius_rag.rag.metrics",
    "rag_bundle": "radius_rag.rag.bundle",
    "rag_service": "radius_rag.rag.service",
    "rag_search": "radius_rag.rag.search",
    "rag_proxy": "radius_rag.rag.proxy",
}

FILES = {
    "rag_metrics.py": DEST / "metrics.py",
    "rag_bundle.py": DEST / "bundle.py",
    "rag_service.py": DEST / "service.py",
    "rag_search.py": DEST / "search.py",
    "rag_proxy.py": DEST / "proxy.py",
}

PAT = re.compile(r"\b(from|import)\s+(rag_[a-z_]+)")
RETRIEVAL_PAT = re.compile(r"\b(from|import)\s+(retrieval_[a-z_]+)")

RETRIEVAL_MAP = {
    "retrieval_event": "radius_rag.observability.event",
}

LLM_MAP = {
    "llm_params": "radius_rag.llm.params",
    "llm_client": "radius_rag.llm.client",
}

LLM_PAT = re.compile(r"\b(from|import)\s+(llm_[a-z_]+)")


def rewrite(text: str) -> str:
    def repl_rag(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = MAP.get(mod)
        if not new:
            return match.group(0)
        if match.group(1) == "from":
            return f"from {new}"
        return f"import {new}"

    def repl_retrieval(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = RETRIEVAL_MAP.get(mod)
        if not new:
            return match.group(0)
        if match.group(1) == "from":
            return f"from {new}"
        return f"import {new}"

    def repl_llm(match: re.Match[str]) -> str:
        mod = match.group(2)
        new = LLM_MAP.get(mod)
        if not new:
            return match.group(0)
        if match.group(1) == "from":
            return f"from {new}"
        return f"import {new}"

    text = PAT.sub(repl_rag, text)
    text = RETRIEVAL_PAT.sub(repl_retrieval, text)
    return LLM_PAT.sub(repl_llm, text)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    init = DEST / "__init__.py"
    if not init.exists():
        init.write_text('"""RAG index, retrieval, and bundle management."""\n', encoding="utf-8")

    for src_name, dest in FILES.items():
        src_path = ROOT / src_name
        if not src_path.exists():
            raise SystemExit(f"missing {src_path}")
        text = src_path.read_text(encoding="utf-8")
        if "radius_rag.rag" in text and src_path.stat().st_size < 500:
            raise SystemExit(
                f"{src_name} looks like a shim already; restore full module before migrate"
            )
        dest.write_text(rewrite(text), encoding="utf-8")

    print(f"wrote {len(FILES)} modules under {DEST}")


if __name__ == "__main__":
    main()
