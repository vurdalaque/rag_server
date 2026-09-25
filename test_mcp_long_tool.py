"""Spawns an isolated pytest process for the long MCP tool integration test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_mcp_generate_image_waits_for_slow_backend() -> None:
    worker = Path(__file__).with_name("test_mcp_long_tool_worker.py")
    env = os.environ.copy()
    env.setdefault("MCP_STATELESS_HTTP", "false")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(worker),
            "-q",
            "--tb=short",
        ],
        cwd=str(Path(__file__).parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        f"worker stdout:\n{completed.stdout}\nworker stderr:\n{completed.stderr}"
    )
