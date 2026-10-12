#!/usr/bin/env bash
# Start RAG stack in tmux (server + optional retrieval evaluator).
#
#   cp env.example .env && pip install -e .
#   ./start.sh
#   ./stop.sh
#   tmux attach -t radius-server
#
# Foreground server (no tmux): ./start.sh --foreground  or  ./start_server.sh

set -euo pipefail

# shellcheck source=scripts/start_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/start_lib.sh"

start_lib_root
start_lib_parse_stack_args "$@"
start_lib_load_dotenv "$START_ENV_FILE"

if [[ "$START_STACK_MODE" == "foreground" ]]; then
    exec "$START_ROOT/start_server.sh" --env-file "$START_ENV_FILE"
fi

if [[ "$START_STACK_MODE" != "evaluator-only" ]]; then
    start_lib_require_admin_token
fi

SERVER_SESSION="${RAG_TMUX_SESSION_SERVER:-radius-server}"
EVAL_SESSION="${RAG_TMUX_SESSION_EVALUATOR:-radius-evaluator}"
start_tmux_session() {
    local name="$1"
    local script="$2"
    local quoted_env
    quoted_env="$(printf '%q' "$START_ENV_FILE")"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "tmux session already running: $name"
        return 0
    fi
    tmux new -d -s "$name" -c "$START_ROOT" "$script --env-file $quoted_env"
    echo "started tmux session: $name"
}

if [[ "$START_STACK_MODE" == "both" || "$START_STACK_MODE" == "server" ]]; then
    start_tmux_session "$SERVER_SESSION" "./start_server.sh"
fi

if [[ "$START_STACK_MODE" == "both" || "$START_STACK_MODE" == "evaluator" ]]; then
    if [[ -z "${RETRIEVAL_DATABASE_URL:-}" ]]; then
        echo "Skipping evaluator tmux session: RETRIEVAL_DATABASE_URL is not set." >&2
    else
        start_tmux_session "$EVAL_SESSION" "./start_evaluator.sh"
    fi
fi

case "${START_ATTACH:-}" in
    server)
        exec tmux attach -t "$SERVER_SESSION"
        ;;
    evaluator|eval)
        exec tmux attach -t "$EVAL_SESSION"
        ;;
    all)
        exec tmux attach -t "$SERVER_SESSION"
        ;;
esac

echo ""
echo "Attach:  tmux attach -t $SERVER_SESSION"
if [[ -n "${RETRIEVAL_DATABASE_URL:-}" ]]; then
    echo "         tmux attach -t $EVAL_SESSION"
fi
echo "List:    tmux ls"
