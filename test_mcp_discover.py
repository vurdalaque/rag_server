from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

import rag_server

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

DISCOVER_BODY = {
    "jsonrpc": "2.0",
    "id": "discover-1",
    "method": "server/discover",
    "params": {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {
                "name": "chatgpt",
                "version": "1.0.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        },
    },
}

INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
}


@pytest.fixture(scope="module")
def mcp_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """MCP session_manager.run() is single-use per MCPServer instance."""
    tmp_path = tmp_path_factory.mktemp("mcp")
    previous_env = {
        key: os.environ.get(key)
        for key in (
            "RAG_ADMIN_TOKEN",
            "RAG_STAGING_DIR",
            "RAG_BUNDLE_STATE_DIR",
            "RAG_INDEX_FILE",
            "RAG_METADATA_FILE",
            "RAG_METRICS_PROBE_ENABLED",
        )
    }

    os.environ["RAG_ADMIN_TOKEN"] = "test-token"
    os.environ["RAG_STAGING_DIR"] = str(tmp_path / "staging")
    os.environ["RAG_BUNDLE_STATE_DIR"] = str(tmp_path / "state")
    os.environ["RAG_INDEX_FILE"] = str(tmp_path / "missing.faiss")
    os.environ["RAG_METADATA_FILE"] = str(tmp_path / "missing.jsonl")
    os.environ["RAG_METRICS_PROBE_ENABLED"] = "false"
    rag_server.rag_service.clear()

    client = TestClient(rag_server.app, base_url="http://localhost:8000")
    client.__enter__()

    yield client

    client.__exit__(None, None, None)

    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_mcp_server_discover_returns_modern_capabilities(
    mcp_client: TestClient,
) -> None:
    response = mcp_client.post(
        "/mcp/",
        json=DISCOVER_BODY,
        headers={
            **MCP_HEADERS,
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": "server/discover",
        },
    )

    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "discover-1"
    result = payload["result"]
    assert "capabilities" in result
    assert "tools" in result["capabilities"]
    assert "2026-07-28" in result.get("supportedVersions", [])
    server_info = result.get("_meta", {}).get(
        "io.modelcontextprotocol/serverInfo",
        {},
    )
    assert server_info.get("name") == "Project Knowledge Gateway"
    json.loads(response.content)


def test_mcp_server_discover_without_routing_headers(
    mcp_client: TestClient,
) -> None:
    """Proxies may strip MCP-Protocol-Version; body _meta must still work."""
    response = mcp_client.post(
        "/mcp/",
        json=DISCOVER_BODY,
        headers=MCP_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert "result" in payload
    assert payload["result"]["supportedVersions"] == ["2026-07-28"]

    init_response = mcp_client.post(
        "/mcp/",
        json=INITIALIZE_BODY,
        headers=MCP_HEADERS,
    )
    assert init_response.status_code == 200
    init_payload = init_response.json()
    assert init_payload["result"]["protocolVersion"] == "2024-11-05"

    mcp_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        },
        headers=MCP_HEADERS,
    )

    tools_response = mcp_client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers=MCP_HEADERS,
    )

    assert tools_response.status_code == 200
    tools_payload = tools_response.json()
    tool_names = {tool["name"] for tool in tools_payload["result"]["tools"]}
    assert tool_names == {
        "search_project",
        "web_search",
        "ask_project",
        "ping",
    }
