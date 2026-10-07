from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest
from mcp.server.mcpserver import MCPServer

import image_mcp_tools
from image_generation import GeneratedImage, ImageGenerationResult
from image_mcp_schemas import generate_image_input_json_schema
from image_mcp_tools import (
    GENERATION_MCP_TOOL_NAMES,
    list_mcp_tool_names,
    list_ping_tool_names,
    published_generate_image_input_schema,
    register_image_tools,
    reset_image_mcp_registration,
)


def _sample_capabilities_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": "comfyui",
        "max_images": 4,
        "max_output_bytes": 20971520,
        "max_reference_images": 10,
        "max_reference_bytes": 10485760,
        "inputs": {
            "reference_images": {
                "supported": True,
                "max_count": 10,
                "max_bytes": 10485760,
            },
            "mask": {
                "supported": True,
                "max_count": 1,
                "max_bytes": 10485760,
                "semantics": "soft_region_guidance",
            },
            "sketch": {
                "supported": True,
                "max_count": 1,
                "max_bytes": 10485760,
                "semantics": "composition_guidance",
            },
        },
        "resolution": {"min": 256, "max": 2048},
        "defaults": {
            "resolution": 1024,
            "steps": 25,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
        },
        "safety_validation_enabled": False,
        "safety_enabled": False,
        "samplers": {"sampler_name": ["euler"], "scheduler": ["simple"]},
        "analysis": {"supported": True, "max_images": 10, "max_bytes": 10485760},
        "segmentation": {
            "supported": True,
            "max_bytes": 10485760,
            "max_dimension": 8192,
            "mask_semantics": "selected_white_1_not_selected_black_0",
            "modes": {
                "text": True,
                "points": True,
                "box": True,
                "mask_refinement": True,
            },
        },
        "upscale": {
            "supported": True,
            "max_input_bytes": 10485760,
            "max_output_bytes": 20971520,
            "max_input_dimension": 4096,
            "max_output_dimension": 8192,
            "supports_scale_factor": True,
            "supports_target_dimensions": True,
            "max_scale": 4.0,
            "default_scale": 2.0,
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _reset_image_tools_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    reset_image_mcp_registration()
    yield
    reset_image_mcp_registration()


def test_list_mcp_tool_names_without_image_tools() -> None:
    assert list_mcp_tool_names(include_image_tools=False) == [
        "search_project",
        "web_search",
        "ask_project",
        "ping",
    ]
    assert list_ping_tool_names() == [
        "search_project",
        "web_search",
        "ask_project",
    ]


def test_list_mcp_tool_names_with_image_tools() -> None:
    server = MCPServer("image-test-list-names")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    register_image_tools(server, backend)

    names = list_mcp_tool_names(include_image_tools=True)
    assert set(GENERATION_MCP_TOOL_NAMES) <= set(names)
    assert "upscale_image" in names


def test_register_image_tools_adds_tools_when_backend_available() -> None:
    server = MCPServer("image-test")
    backend = MagicMock()
    backend.capabilities = AsyncMock(
        return_value={
            "backend": "comfyui",
            "max_images": 4,
        }
    )

    register_image_tools(server, backend)
    assert image_mcp_tools.image_tools_registered()

    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert tool_names == set(GENERATION_MCP_TOOL_NAMES) | {"upscale_image"}


def test_register_image_tools_is_idempotent() -> None:
    server = MCPServer("image-test-idempotent")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})

    register_image_tools(server, backend)
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    assert len(tools) == len(GENERATION_MCP_TOOL_NAMES) + 1
    assert "upscale_image" in {tool.name for tool in tools}
    generate_tool = next(tool for tool in tools if tool.name == "generate_image")
    assert "mask" in generate_tool.input_schema.get("properties", {})
    assert "sketch" in generate_tool.input_schema.get("properties", {})


def test_generate_image_tool_returns_error_without_backend() -> None:
    server = MCPServer("image-test-missing-backend")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    register_image_tools(server, backend)

    reset_image_mcp_registration()
    image_mcp_tools._registered = True
    image_mcp_tools._backend = None

    result = asyncio.run(server.call_tool("generate_image", {"prompt": "a red cube"}))
    assert result.is_error is True
    text = next(block for block in result.content if block.type == "text")
    assert json.loads(text.text)["error"]["code"] == "backend_unavailable"


def test_upscale_tool_stays_visible_and_reports_unavailable_backend() -> None:
    from image_mcp_backends import ImageMcpBackends
    from test_image_contract import _png_b64

    server = MCPServer("image-test-upscale-unavailable")
    register_image_tools(server, None, backends=ImageMcpBackends())
    tools = asyncio.run(server.list_tools())
    assert "upscale_image" in {tool.name for tool in tools}

    capabilities = asyncio.run(server.call_tool("image_generation_capabilities", {}))
    assert capabilities.structured_content["upscale"]["supported"] is False
    result = asyncio.run(
        server.call_tool("upscale_image", {"image": _png_b64(), "scale": 4})
    )
    assert result.is_error is True
    error = json.loads(next(block.text for block in result.content if block.type == "text"))
    assert error["error"]["code"] == "backend_unavailable"


def test_capabilities_tool_delegates_to_backend() -> None:
    server = MCPServer("image-test-capabilities")
    backend = MagicMock()
    payload = _sample_capabilities_payload(max_images=2)
    backend.capabilities = AsyncMock(return_value=payload)
    register_image_tools(server, backend)

    result = asyncio.run(server.call_tool("image_generation_capabilities", {}))
    assert result.is_error is not True
    assert result.structured_content is not None
    assert result.structured_content["max_images"] == 2
    assert result.structured_content["backend"] == "comfyui"
    backend.capabilities.assert_awaited_once()


def test_image_tools_publish_output_schema() -> None:
    server = MCPServer("image-test-output-schema")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    by_name = {tool.name: tool for tool in tools}

    for name in GENERATION_MCP_TOOL_NAMES:
        schema = by_name[name].output_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_capabilities_structured_output_matches_schema() -> None:
    server = MCPServer("image-test-cap-schema")
    backend = MagicMock()
    payload = _sample_capabilities_payload()
    backend.capabilities = AsyncMock(return_value=payload)
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    cap_tool = next(t for t in tools if t.name == "image_generation_capabilities")
    result = asyncio.run(server.call_tool("image_generation_capabilities", {}))

    assert result.structured_content is not None
    jsonschema.validate(instance=result.structured_content, schema=cap_tool.output_schema)
    assert result.structured_content["samplers"]["sampler_name"] == ["euler"]
    assert result.structured_content["inputs"]["mask"]["semantics"] == "soft_region_guidance"
    assert result.structured_content["safety_validation_enabled"] is False
    assert result.structured_content["max_reference_images"] == 10


def test_generate_image_success_structured_output_and_images() -> None:
    server = MCPServer("image-test-generate-structured")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 16
    backend.generate = AsyncMock(
        return_value=ImageGenerationResult(
            images=[
                GeneratedImage(data=png_bytes, mime_type="image/png", filename="a.png"),
                GeneratedImage(data=png_bytes, mime_type="image/png", filename="b.png"),
            ],
            seed=42,
            prompt_id="prompt-1",
            timings={"generate_ms": 1250.0},
        )
    )
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    gen_tool = next(t for t in tools if t.name == "generate_image")
    result = asyncio.run(server.call_tool("generate_image", {"prompt": "red cube"}))

    assert result.is_error is not True
    assert all(block.type == "text" for block in result.content)
    assert result.structured_content is not None
    assert "data" not in result.structured_content
    assert result.structured_content["seed"] == 42
    assert result.structured_content["image_count"] == 2
    assert result.structured_content["timings"]["total_ms"] >= 0
    assert result.structured_content["timings"]["generate_ms"] == 1250.0
    artifacts = result.structured_content["artifacts"]
    assert [artifact["artifact_id"] for artifact in artifacts] == ["prompt-1_0", "prompt-1_1"]
    assert all(artifact["mime_type"] == "image/png" for artifact in artifacts)
    assert all(artifact["size_bytes"] == len(png_bytes) for artifact in artifacts)
    jsonschema.validate(instance=result.structured_content, schema=gen_tool.output_schema)
    text_blocks = [block for block in result.content if block.type == "text"]
    assert len(text_blocks) == 1


def test_generate_image_comfy_error_stays_is_error() -> None:
    from image_generation_errors import ImageGenerationTimeoutError

    server = MCPServer("image-test-generate-error")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    backend.generate = AsyncMock(
        side_effect=ImageGenerationTimeoutError("timed out", timeout_seconds=30.0)
    )
    register_image_tools(server, backend)

    result = asyncio.run(server.call_tool("generate_image", {"prompt": "cube"}))
    assert result.is_error is True
    assert result.structured_content is None
    text = next(block for block in result.content if block.type == "text")
    body = json.loads(text.text)
    assert body["error"]["code"] == "timeout"


def test_generate_image_tool_schema_is_model_agnostic() -> None:
    server = MCPServer("image-test-schema")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value={})
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    generate_tool = next(tool for tool in tools if tool.name == "generate_image")
    schema: dict[str, Any] = generate_tool.input_schema
    properties = schema.get("properties", {})
    assert "prompt" in properties
    assert "sampler" in properties
    assert "scheduler" in properties
    assert "mask" in properties
    assert "sketch" in properties
    assert "unet_name" not in properties


def test_generate_image_tools_list_input_schema_matches_canonical_model() -> None:
    server = MCPServer("image-test-list-schema")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    register_image_tools(server, backend)

    tools = asyncio.run(server.list_tools())
    generate_tool = next(tool for tool in tools if tool.name == "generate_image")
    published = generate_tool.input_schema
    canonical = generate_image_input_json_schema()

    assert published.get("required") == ["prompt"]
    assert "mask" not in published.get("required", [])
    assert "sketch" not in published.get("required", [])
    assert "reference_images" not in published.get("required", [])

    for key in ("prompt", "reference_images", "mask", "sketch"):
        assert key in published.get("properties", {})
        assert key in canonical.get("properties", {})

    mask_prop = published["properties"]["mask"]
    sketch_prop = published["properties"]["sketch"]
    ref_prop = published["properties"]["reference_images"]
    assert "description" in mask_prop
    assert "WHERE" in mask_prop["description"] or "region" in mask_prop["description"].lower()
    assert "description" in sketch_prop
    assert "HOW" in sketch_prop["description"] or "composition" in sketch_prop["description"].lower()
    assert "description" in ref_prop
    assert "WHAT" in ref_prop["description"] or "visual" in ref_prop["description"].lower()

    assert published_generate_image_input_schema() == canonical
    assert "reference images" in generate_tool.description.lower()
    assert "mask" in generate_tool.description.lower()
    assert "sketch" in generate_tool.description.lower()


def test_generate_image_tools_list_via_mcp_jsonrpc() -> None:
    """tools/list over MCPServer (same path as HTTP /mcp after initialize)."""
    server = MCPServer("image-test-jsonrpc-list")
    backend = MagicMock()
    backend.capabilities = AsyncMock(return_value=_sample_capabilities_payload())
    register_image_tools(server, backend)

    listed = asyncio.run(server.list_tools())
    names = {tool.name for tool in listed}
    assert "generate_image" in names
    schema = next(t for t in listed if t.name == "generate_image").input_schema
    assert set(schema.get("properties", {})) >= {
        "prompt",
        "reference_images",
        "mask",
        "sketch",
    }


def test_probe_skipped_when_disabled() -> None:
    from image_generation import ComfyUIBackend
    from image_generation_config import ImageGenerationConfig

    config = ImageGenerationConfig(
        enabled=False,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=__import__("pathlib").Path("."),
        workflow_template_path=__import__("pathlib").Path("template.json"),
        probe_timeout=1.0,
        generate_timeout=1.0,
        upload_timeout_seconds=1.0,
        output_read_timeout_seconds=1.0,
        max_images=1,
        max_output_bytes=1024,
        max_reference_images=0,
        max_reference_bytes=1024,
        safety_enabled=False,
        safety_fail_closed=True,
        default_resolution=512,
        min_resolution=256,
        max_resolution=2048,
        default_steps=10,
        min_steps=1,
        max_steps=100,
        default_cfg=1.0,
        min_cfg=0.0,
        max_cfg=30.0,
        default_sampler="euler",
        default_scheduler="simple",
        default_denoise=1.0,
        min_denoise=0.0,
        max_denoise=1.0,
        allowed_input_mime_types=frozenset({"image/png"}),
    )
    backend = ComfyUIBackend(config)

    assert asyncio.run(backend.probe()) is False


def test_generation_job_deduplicates_and_recovers_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    server = MCPServer("durable-generation-test")
    backend = MagicMock()
    backend.generate = AsyncMock(return_value=ImageGenerationResult(
        images=[GeneratedImage(data=b"\x89PNG\r\n\x1a\n" + b"x", mime_type="image/png")],
        seed=1, prompt_id="durable-prompt", timings={},
    ))
    register_image_tools(server, backend)

    async def run():
        first = await server.call_tool("generate_image", {"prompt": "red cube", "request_id": "test-job-1"})
        assert not first.is_error, first
        duplicate = await server.call_tool("generate_image", {"prompt": "red cube", "request_id": "test-job-1"})
        assert not duplicate.is_error, duplicate
        await asyncio.sleep(0.1)
        status = await server.call_tool("get_generation_job", {"request_id": "test-job-1"})
        assert not status.is_error, status
        assert status.structured_content["status"] == "completed"
        assert status.structured_content["result"]["artifacts"][0]["artifact_id"] == "durable-prompt_0"
        conflict = await server.call_tool("generate_image", {"prompt": "blue cube", "request_id": "test-job-1"})
        assert conflict.is_error

    asyncio.run(run())
    backend.generate.assert_awaited_once()


def test_artifact_expiry_preserves_job_database(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    image_mcp_tools._job_register("old-job", "fingerprint", 1)
    stale = tmp_path / "old-prompt_0.png"
    stale.write_bytes(b"\x89PNG\r\n\x1a\n")
    unrelated = tmp_path / "notes.png"
    unrelated.write_bytes(b"x")
    assert image_mcp_tools.cleanup_expired_image_artifacts(now=image_mcp_tools.time.time() + 10**7) == 1
    assert not stale.exists()
    assert unrelated.exists()
    assert image_mcp_tools._job_read("old-job") is not None


def test_delivery_confirmation_removes_only_job_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    image_mcp_tools._job_register("delivered-job", "fingerprint", 2)
    artifacts = []
    for index in range(2):
        artifact = image_mcp_tools.store_generated_image_artifact(
            "delivery-prompt", index,
            GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
        )
        artifacts.append(artifact.model_dump(mode="json"))
    image_mcp_tools._job_update("delivered-job", "completed", result={"artifacts": artifacts})
    untouched = tmp_path / "unrelated-prompt_0.png"
    untouched.write_bytes(b"unrelated")
    ids = [item["artifact_id"] for item in artifacts]
    with pytest.raises(image_mcp_tools.InvalidRequestError):
        image_mcp_tools.confirm_generated_image_delivery("delivered-job", ids[:1])
    assert all((tmp_path / f"{item}.png").exists() for item in ids)
    first = image_mcp_tools.confirm_generated_image_delivery("delivered-job", ids)
    second = image_mcp_tools.confirm_generated_image_delivery("delivered-job", ids)
    assert first["status"] == second["status"] == "delivered"
    assert all(image_mcp_tools.artifact_delivery_confirmed(item) for item in ids)
    assert all(not (tmp_path / f"{item}.png").exists() for item in ids)
    assert untouched.exists()
    assert image_mcp_tools._job_read("delivered-job") is not None


def test_delivery_confirmation_tool_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    server = MCPServer("delivery-confirmation-test")
    register_image_tools(server, MagicMock())
    image_mcp_tools._job_register("mcp-delivery", "fingerprint", 1)
    artifact = image_mcp_tools.store_generated_image_artifact(
        "tool-prompt", 0, GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
    )
    image_mcp_tools._job_update("mcp-delivery", "completed", result={"artifacts": [artifact.model_dump(mode="json")]})

    async def run():
        bad = await server.call_tool("confirm_generation_delivery", {
            "request_id": "mcp-delivery", "artifact_ids": ["wrong_0"],
        })
        assert bad.is_error
        for _ in range(2):
            response = await server.call_tool("confirm_generation_delivery", {
                "request_id": "mcp-delivery", "artifact_ids": [artifact.artifact_id],
            })
            assert not response.is_error
            assert response.structured_content["status"] == "delivered"

    asyncio.run(run())
    assert not (tmp_path / f"{artifact.artifact_id}.png").exists()


def test_delivered_artifact_cannot_be_recovered(tmp_path, monkeypatch) -> None:
    from fastapi import HTTPException
    import rag_server

    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    image_mcp_tools._job_register("delivered-job", "fingerprint", 1)
    artifact = image_mcp_tools.store_generated_image_artifact(
        "delivered-prompt", 0, GeneratedImage(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png"),
    )
    image_mcp_tools._job_update("delivered-job", "completed", result={"artifacts": [artifact.model_dump(mode="json")]})
    image_mcp_tools.confirm_generated_image_delivery("delivered-job", [artifact.artifact_id])
    backend = MagicMock()
    backend.recover_generated_image = AsyncMock()
    monkeypatch.setattr(rag_server, "image_backend", backend)
    with pytest.raises(HTTPException) as error:
        asyncio.run(rag_server.download_mcp_artifact(artifact.artifact_id))
    assert error.value.status_code == 404
    backend.recover_generated_image.assert_not_awaited()


def test_env_example_documents_all_application_variables() -> None:
    root = Path(__file__).resolve().parent
    pattern = re.compile(r'''(?:os\.getenv|os\.environ\.get|env_bool|env_int|env_float|_env_int|_env_float)\(\s*['"]([A-Z][A-Z0-9_]+)['"]''')
    used = set()
    for source in root.glob('*.py'):
        if not source.name.startswith('test_'):
            used.update(pattern.findall(source.read_text(encoding='utf-8')))
    example = (root / 'env.example').read_text(encoding='utf-8')
    documented = set(re.findall(r'(?m)^\s*#?\s*([A-Z][A-Z0-9_]+)\s*=', example))
    assert not used - documented, ', '.join(sorted(used - documented))


def test_generation_job_unknown_is_not_replayed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_GENERATION_TIMEOUT", "0")
    server = MCPServer("unknown-generation-test")
    backend = MagicMock()
    backend.generate = AsyncMock()
    register_image_tools(server, backend)
    job, created = image_mcp_tools._job_register("lost-submit", "fingerprint", 1)
    assert created and job["status"] == "running"
    with image_mcp_tools._jobs_db() as db:
        db.execute("UPDATE generation_jobs SET updated=0 WHERE request_id=?", ("lost-submit",))
    result = asyncio.run(server.call_tool("get_generation_job", {"request_id": "lost-submit"}))
    assert result.structured_content["status"] == "unknown"
    backend.generate.assert_not_awaited()


def test_generation_job_recovers_known_prompt_without_submission(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCP_ARTIFACTS_DIR", str(tmp_path))
    server = MCPServer("recovered-generation-test")
    backend = MagicMock()
    backend.generate = AsyncMock()
    backend.recover_generated_image = AsyncMock(return_value=(
        GeneratedImage(data=b"\x89PNG\r\n\x1a\n" + b"x", mime_type="image/png"), "success",
    ))
    register_image_tools(server, backend)
    image_mcp_tools._job_register("recover-job", "fingerprint", 1)
    image_mcp_tools._job_update("recover-job", "running", prompt_ids=["known-prompt"])
    result = asyncio.run(server.call_tool("get_generation_job", {"request_id": "recover-job"}))
    assert result.structured_content["status"] == "completed"
    assert result.structured_content["result"]["artifacts"][0]["artifact_id"] == "known-prompt_0"
    backend.generate.assert_not_awaited()
