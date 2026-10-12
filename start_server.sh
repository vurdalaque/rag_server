#!/usr/bin/env bash
# RAG HTTP/MCP server — see also start.sh (tmux stack).
set -euo pipefail

# shellcheck source=scripts/start_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/start_lib.sh"

start_lib_root
start_lib_parse_server_args "$@"
start_lib_load_dotenv "$START_ENV_FILE"
start_lib_require_admin_token
start_lib_select_python
start_lib_preflight_python

if ! start_lib_run_python -c "import uvicorn" >/dev/null 2>&1; then
    echo "uvicorn is not installed." >&2
    start_lib_install_hint
    exit 1
fi

if ! start_lib_run_python -c "import rag_server; import radius_rag" >/dev/null 2>&1; then
    echo "Cannot import rag_server or radius_rag." >&2
    start_lib_install_hint
    exit 1
fi

if start_lib_retrieval_telemetry_wanted; then
    start_lib_require_asyncpg_if_db
fi

start_lib_export_server_defaults
start_lib_exec_uvicorn
