#!/usr/bin/env bash
# Smoke MCP on a running rag_server (stateless Streamable HTTP, JSON body on POST).
# Usage:
#   BASE_URL=http://192.168.10.250:8000 ./tests/mcp_smoke.sh
#   CURL_MAX=1200 ./tests/mcp_smoke.sh   # real generate_image needs the full timeout
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MCP_URL="${BASE_URL%/}/mcp/"
CURL_MAX="${CURL_MAX:-1200}"

hdr_accept='Accept: application/json, text/event-stream'
hdr_json='Content-Type: application/json'

echo "== health =="
curl -fsS "${BASE_URL}/health" | head -c 400
echo

echo "== initialize =="
init_out="$(mktemp)"
init_headers="$(mktemp)"
curl -fsS -D "${init_headers}" -o "${init_out}" \
  -H "${hdr_accept}" -H "${hdr_json}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  "${MCP_URL}"
session_id="$(grep -i '^mcp-session-id:' "${init_headers}" | head -1 | cut -d' ' -f2 | tr -d '\r')"
if [[ -z "${session_id}" ]]; then
  echo "NOTE: no mcp-session-id (expected for Streamable HTTP); continuing without session header."
  session_hdr=()
else
  echo "session_id=${session_id}"
  session_hdr=(-H "mcp-session-id: ${session_id}")
fi
cat "${init_out}"
echo

echo "== notifications/initialized =="
curl -fsS -o /dev/null -w '%{http_code}\n' \
  -H "${hdr_accept}" -H "${hdr_json}" "${session_hdr[@]}" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' \
  "${MCP_URL}"

echo "== ping =="
curl -fsS \
  -H "${hdr_accept}" -H "${hdr_json}" "${session_hdr[@]}" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ping","arguments":{}}}' \
  "${MCP_URL}"
echo

echo "== generate_image (may take up to ${CURL_MAX}s) =="
curl -fsS -m "${CURL_MAX}" -w '\nhttp_code=%{http_code} time_total=%{time_total}s\n' \
  -H "${hdr_accept}" -H "${hdr_json}" "${session_hdr[@]}" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a red cube on white background"}}}' \
  "${MCP_URL}" | tee /tmp/mcp_generate_smoke.json
echo

if grep -q '"Connection closed"' /tmp/mcp_generate_smoke.json 2>/dev/null; then
  echo "FAIL: MCP returned Connection closed"
  exit 1
fi
echo "OK"
