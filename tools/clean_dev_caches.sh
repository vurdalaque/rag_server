#!/usr/bin/env bash
# Remove regenerable dev caches under rag_server (not .venv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for name in .pytest_cache .pytest_tmp .test_metrics_run .tmp .mypy_cache .ruff_cache; do
    if [[ -e "$name" ]]; then
        rm -rf "$name"
        echo "removed $name"
    fi
done
