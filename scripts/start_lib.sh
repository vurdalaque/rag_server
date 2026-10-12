# Shared helpers for start_server.sh / start_evaluator.sh (bash).
# shellcheck shell=bash

start_lib_root() {
    # Caller is repo-root start_*.sh (BASH_SOURCE[1]), not this file under scripts/.
    START_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    cd "$START_ROOT"
    export PYTHONPATH="${START_ROOT}/src:${START_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

start_lib_usage_server() {
    cat <<'EOF'
start_server.sh — RAG HTTP/MCP server (uvicorn rag_server:app)

  ./start_server.sh [--env-file .env] [--host HOST] [--port PORT]
EOF
}

start_lib_usage_evaluator() {
    cat <<'EOF'
start_evaluator.sh — retrieval LLM judge worker

  ./start_evaluator.sh [--env-file .env]
EOF
}

start_lib_usage_stack() {
    cat <<'EOF'
start.sh — tmux: RAG server (+ optional retrieval evaluator)

  ./start.sh [--env-file .env]
  ./start.sh --server-only | --evaluator-only
  ./start.sh --foreground          # server in current shell (no tmux)
  ./start.sh --attach server|evaluator|all

Direct (no tmux): ./start_server.sh  ./start_evaluator.sh
EOF
}

start_lib_usage_stop() {
    cat <<'EOF'
stop.sh — stop tmux sessions started by ./start.sh

  ./stop.sh [--env-file .env]
  ./stop.sh --server-only | --evaluator-only
EOF
}

start_lib_parse_server_args() {
    START_ENV_FILE=".env"
    START_HOST=""
    START_PORT=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                start_lib_usage_server
                exit 0
                ;;
            --env-file)
                START_ENV_FILE="$2"
                shift 2
                ;;
            --host)
                START_HOST="$2"
                shift 2
                ;;
            --port)
                START_PORT="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1" >&2
                exit 1
                ;;
        esac
    done
}

start_lib_parse_evaluator_args() {
    START_ENV_FILE=".env"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                start_lib_usage_evaluator
                exit 0
                ;;
            --env-file)
                START_ENV_FILE="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1" >&2
                exit 1
                ;;
        esac
    done
}

start_lib_parse_stack_args() {
    START_ENV_FILE=".env"
    START_STACK_MODE="both"
    START_ATTACH=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                start_lib_usage_stack
                exit 0
                ;;
            --env-file)
                START_ENV_FILE="$2"
                shift 2
                ;;
            --server-only)
                START_STACK_MODE="server"
                shift
                ;;
            --evaluator-only)
                START_STACK_MODE="evaluator"
                shift
                ;;
            --foreground)
                START_STACK_MODE="foreground"
                shift
                ;;
            --attach)
                START_ATTACH="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1 (try --help)" >&2
                exit 1
                ;;
        esac
    done
}

start_lib_parse_stop_args() {
    START_ENV_FILE=".env"
    START_STACK_MODE="both"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                start_lib_usage_stop
                exit 0
                ;;
            --env-file)
                START_ENV_FILE="$2"
                shift 2
                ;;
            --server-only)
                START_STACK_MODE="server"
                shift
                ;;
            --evaluator-only)
                START_STACK_MODE="evaluator"
                shift
                ;;
            *)
                echo "Unknown argument: $1 (try --help)" >&2
                exit 1
                ;;
        esac
    done
}

start_lib_load_dotenv() {
    local env_file="$1"
    if [[ ! -f "$env_file" ]]; then
        echo "Env file not found: $env_file" >&2
        echo "Copy env.example to .env and set RAG_ADMIN_TOKEN." >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
}

start_lib_require_admin_token() {
    if [[ -z "${RAG_ADMIN_TOKEN:-}" ]]; then
        echo "RAG_ADMIN_TOKEN is not set in $START_ENV_FILE." >&2
        exit 1
    fi
}

start_lib_activate_conda_env() {
    local env_name="$1"
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Cannot find conda to activate env: $env_name" >&2
        return 1
    fi
    conda activate "$env_name"
}

start_lib_activate_venv() {
    local venv_path="$1"
    venv_path="${venv_path/#\~/$HOME}"
    if [[ ! -f "$venv_path/bin/activate" ]]; then
        echo "Venv activate script not found: $venv_path/bin/activate" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
}

start_lib_find_uv() {
    START_UV_BIN=""
    if command -v uv >/dev/null 2>&1; then
        START_UV_BIN="$(command -v uv)"
        return 0
    fi
    for candidate in \
        "${HOME}/.local/bin/uv" \
        "${HOME}/.cargo/bin/uv" \
        "/usr/local/bin/uv"
    do
        if [[ -x "$candidate" ]]; then
            START_UV_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

start_lib_select_python() {
    START_USE_UV=false
    START_PYTHON_BIN="${PYTHON:-python}"

    if [[ -n "${RAG_VENV:-}" ]]; then
        start_lib_activate_venv "$RAG_VENV"
        START_PYTHON_BIN="$(command -v python)"
    elif [[ -n "${RAG_CONDA_ENV:-}" ]]; then
        start_lib_activate_conda_env "$RAG_CONDA_ENV"
        START_PYTHON_BIN="$(command -v python)"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        START_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "$START_ROOT/.venv/bin/python" ]]; then
        START_PYTHON_BIN="$START_ROOT/.venv/bin/python"
        # shellcheck disable=SC1091
        source "$START_ROOT/.venv/bin/activate"
    else
        case "${RAG_USE_UV:-auto}" in
            1|true|yes|on)
                if start_lib_find_uv; then
                    START_USE_UV=true
                else
                    echo "RAG_USE_UV is set but uv was not found in PATH." >&2
                    exit 1
                fi
                ;;
            auto)
                if [[ -f "$START_ROOT/pyproject.toml" ]] && start_lib_find_uv; then
                    START_USE_UV=true
                fi
                ;;
        esac
    fi
}

start_lib_run_python() {
    if [[ "$START_USE_UV" == true ]]; then
        if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -d "${HOME}/.venv" ]]; then
            "$START_UV_BIN" run --active python "$@"
        else
            "$START_UV_BIN" run python "$@"
        fi
    else
        "$START_PYTHON_BIN" "$@"
    fi
}

start_lib_install_hint() {
    local exe
    exe="$(start_lib_run_python -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$START_PYTHON_BIN")"
    echo "Install into the same interpreter the start scripts use:" >&2
    echo "  $exe -m pip install -e ." >&2
    if [[ "$START_USE_UV" == true ]]; then
        echo "  # or: cd \"$START_ROOT\" && uv pip install -e . --python \"$exe\"" >&2
    fi
}

start_lib_retrieval_telemetry_wanted() {
    case "${RETRIEVAL_TELEMETRY_ENABLED:-false}" in
        1|true|yes|on|TRUE|YES|ON) return 0 ;;
    esac
    return 1
}

start_lib_require_asyncpg_if_db() {
    if [[ -z "${RETRIEVAL_DATABASE_URL:-}" ]]; then
        return 0
    fi
    if start_lib_run_python -c "import asyncpg" >/dev/null 2>&1; then
        return 0
    fi
    echo "RETRIEVAL_DATABASE_URL is set but asyncpg is not installed." >&2
    echo "Python: $(start_lib_run_python -c 'import sys; print(sys.executable)')" >&2
    echo "Set RAG_VENV in .env to the venv where you ran pip install -e ." >&2
    start_lib_install_hint
    exit 1
}

start_lib_preflight_python() {
    if [[ "$START_USE_UV" != true ]]; then
        if ! command -v "$START_PYTHON_BIN" >/dev/null 2>&1; then
            echo "Python not found: $START_PYTHON_BIN" >&2
            echo "Set RAG_VENV=~/.venv in .env for tmux/non-interactive shells." >&2
            exit 1
        fi
    fi
}

start_lib_export_server_defaults() {
    export RAG_HOST="${START_HOST:-${RAG_HOST:-0.0.0.0}}"
    export RAG_PORT="${START_PORT:-${RAG_PORT:-8000}}"
    export EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8002/v1/embeddings}"
    export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding}"
    export RAG_STAGING_DIR="${RAG_STAGING_DIR:-./data/staging}"
    export RAG_BUNDLE_STATE_DIR="${RAG_BUNDLE_STATE_DIR:-./data/state}"
    mkdir -p "$RAG_STAGING_DIR" "$RAG_BUNDLE_STATE_DIR"
}

start_lib_exec_uvicorn() {
    local -a uvicorn_args=(
        -m uvicorn rag_server:app
        --host "$RAG_HOST"
        --port "$RAG_PORT"
        --timeout-graceful-shutdown 30
    )
    if [[ "$START_USE_UV" == true ]]; then
        echo "Runner:  $START_UV_BIN run python"
        echo "URL:     http://${RAG_HOST}:${RAG_PORT}"
        if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -d "${HOME}/.venv" ]]; then
            exec "$START_UV_BIN" run --active python "${uvicorn_args[@]}"
        fi
        exec "$START_UV_BIN" run python "${uvicorn_args[@]}"
    fi
    echo "Python:  $("$START_PYTHON_BIN" -c 'import sys; print(sys.executable)')"
    echo "Starting RAG server on http://${RAG_HOST}:${RAG_PORT}"
    echo "Staging: ${RAG_STAGING_DIR}"
    echo "State:   ${RAG_BUNDLE_STATE_DIR}"
    echo "Embedding: ${EMBEDDING_URL} (${EMBEDDING_MODEL})"
    exec "$START_PYTHON_BIN" "${uvicorn_args[@]}"
}

start_lib_exec_evaluator() {
    if [[ -z "${RETRIEVAL_DATABASE_URL:-}" ]]; then
        echo "RETRIEVAL_DATABASE_URL is required for the evaluator." >&2
        exit 1
    fi
    start_lib_require_asyncpg_if_db
    echo "Python:  $(start_lib_run_python -c 'import sys; print(sys.executable)')"
    echo "Evaluator: retrieval_obs (RETRIEVAL_DATABASE_URL set)"
    if [[ "$START_USE_UV" == true ]]; then
        if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -d "${HOME}/.venv" ]]; then
            exec "$START_UV_BIN" run --active python -m retrieval_evaluator
        fi
        exec "$START_UV_BIN" run python -m retrieval_evaluator
    fi
    exec "$START_PYTHON_BIN" -m retrieval_evaluator
}
