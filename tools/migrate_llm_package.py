#!/usr/bin/env python3
"""Copy llm_* modules into src/radius_rag/llm/."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "radius_rag" / "llm"

MAP = {
    "llm_params": "radius_rag.llm.params",
    "llm_client": "radius_rag.llm.client",
}

FILES = {
    "llm_params.py": DEST / "params.py",
    "llm_client.py": DEST / "client.py",
}

PAT = re.compile(r"\b(from|import)\s+(llm_[a-z_]+)")


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
    DEST.mkdir(parents=True, exist_ok=True)
    init = DEST / "__init__.py"
    if not init.exists():
        init.write_text('"""LLM HTTP client and request defaults."""\n', encoding="utf-8")

    for src_name, dest in FILES.items():
        src_path = ROOT / src_name
        text = src_path.read_text(encoding="utf-8")
        if "radius_rag.llm" in text and len(text) < 400:
            raise SystemExit(f"{src_name} looks like a shim; restore full module first")
        dest.write_text(rewrite(text), encoding="utf-8")

    print(f"wrote {len(FILES)} modules under {DEST}")


if __name__ == "__main__":
    main()
