from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import comfy_client
import comfy_workflow
from comfy_client import ComfyUIClient
from image_generation_config import _DEFAULT_TEMPLATE, ImageGenerationConfig
from comfy_workflow import (
    WorkflowBuildParams,
    build_image_workflow,
    load_workflow_template,
    resolve_comfy_output_path,
    validate_generation_params,
)
from image_generation_errors import (
    ImageGenerationTimeoutError,
    OutputInvalidError,
    TooManyImagesError,
    UnsupportedParameterError,
)

SAMPLERS = frozenset({"euler", "heun"})
SCHEDULERS = frozenset({"simple", "normal"})


@pytest.fixture
def config(tmp_path: Path) -> ImageGenerationConfig:
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(load_workflow_template(_DEFAULT_TEMPLATE)),
        encoding="utf-8",
    )
    return ImageGenerationConfig(
        enabled=True,
        comfy_base_url="http://127.0.0.1:8188",
        comfyui_output_root=tmp_path / "output",
        workflow_template_path=template,
        probe_timeout=5.0,
        generate_timeout=30.0,
        upload_timeout_seconds=5.0,
        output_read_timeout_seconds=5.0,
        max_images=4,
        max_output_bytes=20 * 1024 * 1024,
        max_reference_images=10,
        max_reference_bytes=1024 * 1024,
        safety_enabled=False,
        safety_fail_closed=True,
        default_resolution=1024,
        min_resolution=256,
        max_resolution=2048,
        default_steps=25,
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


def _build(
    config: ImageGenerationConfig,
    *,
    request_id: str = "req-1",
    images: tuple[str, ...] = (),
    seed: int | None = 42,
    **kwargs: object,
) -> comfy_workflow.WorkflowBuildResult:
    params = WorkflowBuildParams(
        request_id=request_id,
        prompt="hello",
        negative_prompt="bad",
        seed=seed,
        input_image_names=images,
        **kwargs,
    )
    return build_image_workflow(
        params,
        config,
        allowed_samplers=SAMPLERS,
        allowed_schedulers=SCHEDULERS,
    )


def test_text_to_image_zero_images(config: ImageGenerationConfig) -> None:
    result = _build(config, seed=99, images=())
    wf = result.workflow
    assert result.seed_used == 99
    assert "9" not in wf and "10" not in wf
    assert all(
        not key.startswith("images.image_")
        for key in wf["4"]["inputs"]
    )
    assert wf["4"]["inputs"]["prompt"] == "hello"
    assert wf["7"]["inputs"]["filename_prefix"] == "mcp/req-1/result"


def test_one_input_image_wiring(config: ImageGenerationConfig) -> None:
    result = _build(config, images=("mcp/req-1/image_1.png",))
    inputs = result.workflow["4"]["inputs"]
    assert "images.image_1" in inputs
    assert "images.image_2" not in inputs
    load_id = inputs["images.image_1"][0]
    assert result.workflow[load_id]["class_type"] == "LoadImage"
    assert result.workflow[load_id]["inputs"]["image"] == "mcp/req-1/image_1.png"


def test_multiple_input_images_wiring(config: ImageGenerationConfig) -> None:
    names = tuple(f"mcp/req-1/image_{i}.png" for i in range(1, 4))
    result = _build(config, images=names)
    inputs = result.workflow["4"]["inputs"]
    for index in range(1, 4):
        key = f"images.image_{index}"
        assert key in inputs
        load_id = inputs[key][0]
        assert result.workflow[load_id]["inputs"]["image"] == names[index - 1]


def test_ten_input_images(config: ImageGenerationConfig) -> None:
    names = tuple(f"img_{i}.png" for i in range(10))
    result = _build(config, images=names)
    wired = [k for k in result.workflow["4"]["inputs"] if k.startswith("images.image_")]
    assert len(wired) == 10


def test_eleven_images_rejected_before_build(config: ImageGenerationConfig) -> None:
    with pytest.raises(TooManyImagesError):
        WorkflowBuildParams(
            request_id="req-1",
            prompt="x",
            input_image_names=tuple(f"i{n}.png" for n in range(11)),
        )
    with pytest.raises(TooManyImagesError):
        validate_generation_params(
            config,
            resolution=1024,
            steps=25,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            input_image_count=11,
        )


def test_dynamic_image_n_keys_are_sequential(config: ImageGenerationConfig) -> None:
    result = _build(config, images=("a.png", "b.png"))
    keys = sorted(
        k for k in result.workflow["4"]["inputs"] if k.startswith("images.image_")
    )
    assert keys == ["images.image_1", "images.image_2"]


def test_deep_copy_prevents_cross_request_mutation(config: ImageGenerationConfig) -> None:
    template = load_workflow_template(config.workflow_template_path)
    first = build_image_workflow(
        WorkflowBuildParams(request_id="a", prompt="one"),
        config,
        template=template,
    )
    second = build_image_workflow(
        WorkflowBuildParams(request_id="b", prompt="two"),
        config,
        template=template,
    )
    first.workflow["4"]["inputs"]["prompt"] = "mutated"
    assert second.workflow["4"]["inputs"]["prompt"] == "two"
    assert template["4"]["inputs"]["prompt"] == ""


def test_randomized_seed_generated(config: ImageGenerationConfig) -> None:
    result = _build(config, seed=None)
    assert result.seed_used >= 0


def test_explicit_seed_preserved(config: ImageGenerationConfig) -> None:
    result = _build(config, seed=123456789)
    assert result.seed_used == 123456789
    assert result.workflow["5"]["inputs"]["seed"] == 123456789


def test_sampler_scheduler_range_validation(config: ImageGenerationConfig) -> None:
    with pytest.raises(UnsupportedParameterError):
        validate_generation_params(
            config,
            resolution=1024,
            steps=25,
            cfg=1.0,
            sampler_name="unknown",
            scheduler="simple",
            denoise=1.0,
            input_image_count=0,
            allowed_samplers=SAMPLERS,
            allowed_schedulers=SCHEDULERS,
        )
    with pytest.raises(UnsupportedParameterError):
        validate_generation_params(
            config,
            resolution=99999,
            steps=25,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            input_image_count=0,
        )


def test_output_path_traversal_rejected(config: ImageGenerationConfig) -> None:
    root = config.comfyui_output_root
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(OutputInvalidError):
        resolve_comfy_output_path(root, "../../../etc/passwd", "")
    with pytest.raises(OutputInvalidError):
        resolve_comfy_output_path(root, "ok.png", "..", "output")


def test_output_symlink_escape_rejected(config: ImageGenerationConfig) -> None:
    root = config.comfyui_output_root
    root.mkdir(parents=True, exist_ok=True)
    outside = config.comfyui_output_root.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(OutputInvalidError):
        resolve_comfy_output_path(root, "link.png", "", "output")


def test_upload_image_mocked(config: ImageGenerationConfig) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"name": "mcp/u1/image_1.png", "subfolder": "", "type": "input"},
        ),
    )

    async def _run() -> None:
        async with httpx.AsyncClient(
            transport=transport,
            base_url=config.comfy_base_url,
        ) as http:
            client = ComfyUIClient(config, http_client=http)
            uploaded = await client.upload_image("image_1.png", b"png", "image/png")
        assert uploaded.name == "mcp/u1/image_1.png"

    asyncio.run(_run())


def test_ws_ignores_other_prompt_id(config: ImageGenerationConfig) -> None:
    messages = [
        json.dumps(
            {
                "type": "executing",
                "data": {"prompt_id": "other", "node": "5"},
            },
        ),
        json.dumps(
            {
                "type": "execution_success",
                "data": {"prompt_id": "target"},
            },
        ),
    ]

    class FakeWS:
        def __init__(self, _url: str) -> None:
            self._messages = list(messages)

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def recv(self) -> str:
            if not self._messages:
                raise AssertionError("unexpected extra recv")
            return self._messages.pop(0)

    with patch.object(comfy_client.websockets, "connect", FakeWS):
        client = ComfyUIClient(config, http_client=AsyncMock())
        asyncio.run(client.wait_for_prompt_ws("target", "client-1"))


def test_ws_execution_error(config: ImageGenerationConfig) -> None:
    messages = [
        json.dumps(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": "p1",
                    "exception_message": "boom",
                },
            },
        ),
    ]

    class FakeWS:
        def __init__(self, _url: str) -> None:
            self._messages = list(messages)

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def recv(self) -> str:
            return self._messages.pop(0)

    with patch.object(comfy_client.websockets, "connect", FakeWS):
        client = ComfyUIClient(config, http_client=AsyncMock())
        with pytest.raises(comfy_client.ExecutionFailedError):
            asyncio.run(client.wait_for_prompt_ws("p1", "client-1"))


def test_ws_timeout(config: ImageGenerationConfig) -> None:
    class FakeWS:
        def __init__(self, _url: str) -> None:
            pass

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def recv(self) -> str:
            await asyncio.sleep(0.05)
            return json.dumps({"type": "status", "data": {}})

    short_config = dataclasses.replace(config, generate_timeout=0.01)

    with patch.object(comfy_client.websockets, "connect", FakeWS):
        client = ComfyUIClient(short_config, http_client=AsyncMock())
        with pytest.raises(ImageGenerationTimeoutError):
            asyncio.run(client.wait_for_prompt_ws("p1", "client-1"))


def test_run_prompt_happy_path_mocked(config: ImageGenerationConfig) -> None:
    workflow = {"5": {"inputs": {"seed": 7}}, "1": {"class_type": "UNETLoader", "inputs": {}}}
    history = {
        "pid-1": {
            "outputs": {
                "7": {
                    "images": [
                        {
                            "filename": "result_00001.png",
                            "subfolder": "mcp/req-1",
                            "type": "output",
                        },
                    ],
                },
            },
        },
    }

    out_dir = config.comfyui_output_root / "mcp" / "req-1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result_00001.png").write_bytes(b"PNG")

    client = ComfyUIClient(config, http_client=AsyncMock())
    client.submit_prompt = AsyncMock(return_value="pid-1")
    client.wait_for_prompt_ws = AsyncMock()
    client.fetch_history = AsyncMock(return_value=history)

    async def _run() -> None:
        outputs = await client.run_workflow(workflow, request_id="req-1")
        assert outputs[0]["data"] == b"PNG"
        client.wait_for_prompt_ws.assert_awaited_once()

    asyncio.run(_run())
