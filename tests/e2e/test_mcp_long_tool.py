"""Spawns an isolated pytest process for the long MCP tool integration test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_mcp_generate_image_waits_for_slow_backend() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests" / "e2e" / "test_mcp_long_tool_worker.py"
    env = os.environ.copy()
    # Проверяются production-дефолты: stateful transport и JSON-ответы.
    env.pop("MCP_JSON_RESPONSE", None)
    env.pop("MCP_STATELESS_HTTP", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(worker),
            "-q",
            "--tb=short",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"worker stdout:\n{completed.stdout}\nworker stderr:\n{completed.stderr}"
    )
