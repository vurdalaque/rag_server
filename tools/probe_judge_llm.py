#!/usr/bin/env python3
"""Probe judge LLM response shape (run: python tools/probe_judge_llm.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import urllib.request

from radius_rag.observability.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT_V1,
    _parse_judge_json,
    validate_judge_response,
)

URL = "http://spark.home:8001/v1/chat/completions"
MODEL = "Qwen3.8-27B"


def main() -> int:
    user_prompt = (
        "Query:\nWhat is asyncio?\n\n"
        "Retrieved chunks (2):\n\n"
        "Chunk 0: file=asyncio.md repo=docs score=0.91\n"
        "asyncio provides async IO in Python.\n\n"
        "---\n\n"
        "Chunk 1: file=other.md repo=docs score=0.42\n"
        "Unrelated content about databases."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT_V1},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    print("=== raw content (first 2000 chars) ===")
    print(content[:2000])
    print()
    parsed = _parse_judge_json(content)
    pcs = parsed.get("per_chunk_scores")
    print(f"per_chunk_scores type: {type(pcs).__name__}")
    print(f"per_chunk_scores preview: {json.dumps(pcs, ensure_ascii=False)[:500]}")
    try:
        validated = validate_judge_response(parsed, 2)
        print("validate_judge_response: OK")
        print("ordered_relevances:", validated["ordered_relevances"])
    except ValueError as exc:
        print(f"validate_judge_response: FAIL — {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
