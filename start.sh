#!/usr/bin/env bash
# Запуск RAG server с переменными из .env
#
#   cp env.example .env
#   ./start.sh
#
# Параметры:
#   ./start.sh --port 8010
#   ./start.sh --env-file .env.local

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

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${RAG_ADMIN_TOKEN:-}" ]]; then
    echo "RAG_ADMIN_TOKEN is not set. Copy env.example to .env and edit the token." >&2
    exit 1
fi

export RAG_HOST="${HOST:-${RAG_HOST:-0.0.0.0}}"
export RAG_PORT="${PORT:-${RAG_PORT:-8000}}"
export EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8002/v1/embeddings}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding}"
export RAG_STAGING_DIR="${RAG_STAGING_DIR:-./data/staging}"
export RAG_BUNDLE_STATE_DIR="${RAG_BUNDLE_STATE_DIR:-./data/state}"

mkdir -p "$RAG_STAGING_DIR" "$RAG_BUNDLE_STATE_DIR"

echo "Starting RAG server on http://${RAG_HOST}:${RAG_PORT}"
echo "Staging: ${RAG_STAGING_DIR}"
echo "State:   ${RAG_BUNDLE_STATE_DIR}"
echo "Embedding: ${EMBEDDING_URL} (${EMBEDDING_MODEL})"

exec python -m uvicorn rag_server:app --host "$RAG_HOST" --port "$RAG_PORT"
