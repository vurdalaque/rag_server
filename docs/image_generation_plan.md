# Image generation — implementation plan (rag_server)

Based on inspection of `rag_server/` (March 2026).

## Reuse

| Component | Location | Use |
|-----------|----------|-----|
| MCP server | `rag_server.py` — `MCPServer`, `@mcp.tool()`, `lifespan` | Register image tools after probe |
| LLM endpoint | `LLM_URL`, `LLM_MODEL` in `rag_service.py` / `rag_server.py` | Safety validator via new `llm_client.py` |
| LLM defaults | `llm_params.apply_llm_defaults` | Extend with explicit `enable_thinking` override for safety |
| HTTP metrics | `rag_metrics.track_mcp_tool`, `track_dependency`, Histogram/Counter patterns | Image generation metrics |
| Discover / health | `/health`, `ping`, `server/discover` in `rag_server.py` | Dynamic `mcp_tools` list |
| Tests | `test_mcp_discover.py`, `TestClient` + lifespan | Same patterns for image tools |

## Files to add

- `image_generation_config.py` — env parsing (`IMAGE_GENERATION_*`, `COMFYUI_*`)
- `image_generation_errors.py` — typed errors → MCP-friendly codes
- `llm_client.py` — async multimodal chat to `LLM_URL` (no RAG)
- `image_safety.py` — optional pre-check, structured JSON, fail-open/closed
- `comfy_workflow.py` — load template, deep-copy, mutate, LoadImage wiring
- `comfy_client.py` — upload, POST `/prompt`, WebSocket wait by `prompt_id`, history fetch
- `image_generation.py` — `ImageGenerationBackend`, `ComfyUIBackend`, probe, `generate_image` logic
- `image_mcp_tools.py` — `register_image_tools(mcp)` if probe OK
- `data/comfy_image_workflow.template.json` — API workflow (from validated ComfyUI export)
- `test_image_generation.py` — mocked HTTP/WS/LLM (prompt checklist §18)

## Files to modify

- `rag_server.py` — lifespan probe, `register_image_tools`, `/health` + `ping` tool lists
- `llm_params.py` — `override_thinking` / preserve explicit `extra_body` for safety
- `rag_metrics.py` — image counters/histograms (bounded labels)
- `pyproject.toml` / `requirements.txt` — `websockets` if not bundled
- `env.example`, `docs/mcp.md`, `README.md`

## Conditional MCP tools

1. At startup (`lifespan`): if `IMAGE_GENERATION_ENABLED`, run `ComfyUIBackend.probe()` (HTTP `/object_info` + node class checks, no diffusion).
2. On success: `register_image_tools(mcp)` registers `generate_image` and `image_generation_capabilities`.
3. Store module-level `image_backend: ComfyUIBackend | None` for runtime calls.
4. If probe fails: tools absent from `tools/list`; `ping` / `/health` omit them.
5. Runtime call when backend is `None`: structured `backend_unavailable` error.

Dynamic re-registration after init is **not** required (per prompt).

## MCP return type

Inspect installed `mcp` SDK: tool handlers currently return `dict` (JSON text). For images, return `CallToolResult` or SDK-equivalent with `ImageContent` (`type: image`, base64 + mime) plus optional text JSON metadata (seed, timings). Verify against `mcp.types` at implementation time.

## WebSocket correlation

- Connect `ws://{comfy_host}/ws?clientId={uuid}`
- On `POST /prompt`, receive `prompt_id`
- Listen for `executing` / `execution_success` / `execution_error` where `data.prompt_id` matches
- Ignore other prompts; timeout per stage
- After success, `GET /history/{prompt_id}` for output `filename` / `subfolder` / `type`

## Capabilities tool

Merge ComfyUI `/object_info` for `KSampler` (sampler_name, scheduler enums) with server policy (max images, bytes, default 1024/25/euler/simple/cfg 1.0).

## Safety

- `IMAGE_GENERATION_SAFETY_ENABLED` — no per-request bypass
- `llm_client.multimodal_chat(..., thinking=False)` guaranteed not overwritten when `RAG_LLM_FORCE_DEFAULTS=true` (fix `apply_llm_defaults`)
- JSON schema response: `allowed`, `categories`, `reason`

## Workflow template

- Base nodes: UNETLoader, CLIPLoader, VAELoader, TextEncodeQwenImage21, KSampler, VAEDecode, SaveImage
- Per request: 0..N `LoadImage`, wire `images.image_k` on node 4, unique node ids, `filename_prefix=mcp/{request_uuid}/result`
