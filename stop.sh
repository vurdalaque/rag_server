#!/usr/bin/env bash
# Stop RAG stack tmux sessions (see ./start.sh).
#
#   ./stop.sh
#   ./stop.sh --server-only

set -euo pipefail

# shellcheck source=scripts/start_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/start_lib.sh"

start_lib_root
start_lib_parse_stop_args "$@"
start_lib_load_dotenv "$START_ENV_FILE"

SERVER_SESSION="${RAG_TMUX_SESSION_SERVER:-radius-server}"
EVAL_SESSION="${RAG_TMUX_SESSION_EVALUATOR:-radius-evaluator}"

stop_tmux_session() {
    local name="$1"
    if tmux has-session -t "$name" 2>/dev/null; then
        tmux kill-session -t "$name"
        echo "stopped tmux session: $name"
    else
        echo "tmux session not running: $name"
    fi
}

if [[ "$START_STACK_MODE" == "both" || "$START_STACK_MODE" == "server" ]]; then
    stop_tmux_session "$SERVER_SESSION"
fi

if [[ "$START_STACK_MODE" == "both" || "$START_STACK_MODE" == "evaluator" ]]; then
    stop_tmux_session "$EVAL_SESSION"
fi
