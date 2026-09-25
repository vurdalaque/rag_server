"""Shared helpers for HTTP MCP integration tests."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

INITIALIZE_BODY: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
}


def session_headers_from_response(response) -> dict[str, str]:
    headers = dict(MCP_HEADERS)
    session_id = response.headers.get("mcp-session-id")
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def mcp_initialize(client: TestClient) -> tuple[dict[str, str], dict[str, Any]]:
    init_response = client.post(
        "/mcp/",
        json=INITIALIZE_BODY,
        headers=MCP_HEADERS,
    )
    assert init_response.status_code == 200, init_response.text
    init_payload = init_response.json()
    headers = session_headers_from_response(init_response)
    client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        },
        headers=headers,
    )
    return headers, init_payload
