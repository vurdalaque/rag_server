#!/usr/bin/env bash
# Retrieval LLM judge worker — separate process from the API server.
set -euo pipefail

# shellcheck source=scripts/start_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/start_lib.sh"

start_lib_root
start_lib_parse_evaluator_args "$@"
start_lib_load_dotenv "$START_ENV_FILE"
start_lib_select_python
start_lib_preflight_python

if ! start_lib_run_python -c "import radius_rag" >/dev/null 2>&1; then
    echo "Cannot import radius_rag." >&2
    start_lib_install_hint
    exit 1
fi

start_lib_exec_evaluator
