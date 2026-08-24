#!/usr/bin/env bash
# Запуск RAG server с переменными из .env
#
#   cp env.example .env
#   ./start.sh
#
# Параметры:
#   ./start.sh --port 8010
#   ./start.sh --env-file .env.local
#
# tmux (venv/uv не активируются автоматически):
#   tmux new -d -s radius-server './start.sh'

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_FILE=".env"
HOST=""
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE" >&2
    echo "Copy env.example to .env and set RAG_ADMIN_TOKEN." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${RAG_ADMIN_TOKEN:-}" ]]; then
    echo "RAG_ADMIN_TOKEN is not set in $ENV_FILE." >&2
    exit 1
fi

activate_conda_env() {
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

activate_venv() {
    local venv_path="$1"
    venv_path="${venv_path/#\~/$HOME}"

    if [[ ! -f "$venv_path/bin/activate" ]]; then
        echo "Venv activate script not found: $venv_path/bin/activate" >&2
        return 1
    fi

    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
}

if [[ -n "${RAG_VENV:-}" ]]; then
    activate_venv "$RAG_VENV"
elif [[ -n "${RAG_CONDA_ENV:-}" ]]; then
    activate_conda_env "$RAG_CONDA_ENV"
fi

USE_UV=false
case "${RAG_USE_UV:-auto}" in
    1|true|yes|on)
        USE_UV=true
        ;;
    auto)
        if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/pyproject.toml" ]]; then
            USE_UV=true
        fi
        ;;
esac

run_python() {
    if [[ "$USE_UV" == true ]]; then
        uv run python "$@"
    else
        "${PYTHON_BIN}" "$@"
    fi
}

PYTHON_BIN="${PYTHON:-python}"

if [[ "$USE_UV" != true ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Python not found: $PYTHON_BIN" >&2
        echo "Set RAG_VENV, RAG_USE_UV=true, or PYTHON=/path/to/python in .env" >&2
        exit 1
    fi
fi

if ! run_python -c "import uvicorn" >/dev/null 2>&1; then
    echo "uvicorn is not installed." >&2
    if [[ "$USE_UV" == true ]]; then
        echo "Install deps: uv sync --active   # when using ~/.venv"
        echo "            or: uv sync          # for project .venv"
    else
        echo "Install deps: pip install -r requirements.txt" >&2
        run_python -c "import sys; print(sys.executable)" >&2 || true
    fi
    echo "For tmux set RAG_VENV=~/.venv or RAG_USE_UV=true in .env" >&2
    exit 1
fi

if ! run_python -c "import rag_server" >/dev/null 2>&1; then
    echo "Cannot import rag_server." >&2
    run_python -c "import rag_server" 2>&1 || true
    exit 1
fi

export RAG_HOST="${HOST:-${RAG_HOST:-0.0.0.0}}"
export RAG_PORT="${PORT:-${RAG_PORT:-8000}}"
export EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8002/v1/embeddings}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding}"
export RAG_STAGING_DIR="${RAG_STAGING_DIR:-./data/staging}"
export RAG_BUNDLE_STATE_DIR="${RAG_BUNDLE_STATE_DIR:-./data/state}"

mkdir -p "$RAG_STAGING_DIR" "$RAG_BUNDLE_STATE_DIR"

if [[ "$USE_UV" == true ]]; then
    echo "Runner:  uv run python"
    exec uv run python -m uvicorn rag_server:app --host "$RAG_HOST" --port "$RAG_PORT"
fi

echo "Python:  $(run_python -c 'import sys; print(sys.executable)')"
echo "Starting RAG server on http://${RAG_HOST}:${RAG_PORT}"
echo "Staging: ${RAG_STAGING_DIR}"
echo "State:   ${RAG_BUNDLE_STATE_DIR}"
echo "Embedding: ${EMBEDDING_URL} (${EMBEDDING_MODEL})"

exec "$PYTHON_BIN" -m uvicorn rag_server:app --host "$RAG_HOST" --port "$RAG_PORT"
