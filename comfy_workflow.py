"""Build per-request ComfyUI API workflows from the shared template."""

from __future__ import annotations

import copy
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from image_generation_config import ImageGenerationConfig, load_image_generation_config
from image_generation_errors import (
    InvalidRequestError,
    OutputInvalidError,
    TooManyImagesError,
    UnsupportedParameterError,
)

TEXT_ENCODE_NODE_ID = "4"
KSAMPLER_NODE_ID = "5"
SAVE_IMAGE_NODE_ID = "7"

_SKETCH_PROMPT_GUIDANCE = (
    "The attached sketch image provides composition, layout, and rough shape or "
    "color guidance. Treat it as spatial guidance; do not necessarily copy its "
    "visual style."
)
_MASK_PROMPT_GUIDANCE = (
    "The attached mask image marks the soft region where the requested edit should "
    "occur. This is generative region guidance only; pixels outside the mask are "
    "not guaranteed to remain identical."
)

_TEMPLATE_CACHE: dict[Path, dict[str, Any]] = {}


WorkflowImageRole = Literal["reference", "sketch", "mask"]


@dataclass(frozen=True)
class WorkflowImageInput:
    filename: str
    role: WorkflowImageRole = "reference"


@dataclass(frozen=True)
class WorkflowBuildParams:
    """Inputs required to mutate the workflow template for one generation."""

    request_id: str
    prompt: str
    negative_prompt: str = ""
    resolution: int | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    denoise: float | None = None
    input_image_names: tuple[str, ...] = ()
    image_inputs: tuple[WorkflowImageInput, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or "/" in self.request_id or "\\" in self.request_id:
            raise InvalidRequestError("request_id must be a non-empty safe identifier")
        slot_count = len(self.image_inputs) or len(self.input_image_names)
        if slot_count > 12:
            raise TooManyImagesError(
                "At most 12 input images are supported",
                count=slot_count,
                max_images=12,
            )


def resolved_workflow_image_inputs(
    params: WorkflowBuildParams,
) -> tuple[WorkflowImageInput, ...]:
    if params.image_inputs:
        return params.image_inputs
    return tuple(
        WorkflowImageInput(name, "reference") for name in params.input_image_names
    )


def resolve_square_resolution(
    config: ImageGenerationConfig,
    *,
    width: int | None,
    height: int | None,
    resolution: int | None = None,
) -> int:
    """Resolve square workflow resolution from optional width/height."""
    if resolution is not None:
        res = resolution
    elif width is not None and height is not None:
        if width != height:
            raise UnsupportedParameterError(
                "width and height must be equal for the current backend",
                width=width,
                height=height,
                constraint="width_must_equal_height",
            )
        res = width
    elif width is not None:
        res = width
    elif height is not None:
        res = height
    else:
        res = config.default_resolution

    if res < config.min_resolution or res > config.max_resolution:
        raise UnsupportedParameterError(
            "resolution out of allowed range",
            resolution=res,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
        )
    return res


def augment_prompt_for_image_roles(
    prompt: str,
    image_inputs: tuple[WorkflowImageInput, ...],
) -> str:
    parts = [prompt.strip()]
    roles = {item.role for item in image_inputs}
    if "sketch" in roles:
        parts.append(_SKETCH_PROMPT_GUIDANCE)
    if "mask" in roles:
        parts.append(_MASK_PROMPT_GUIDANCE)
    return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True)
class WorkflowBuildResult:
    workflow: dict[str, Any]
    seed_used: int


def load_workflow_template(path: Path) -> dict[str, Any]:
    """Load and cache the API-format workflow template from disk."""
    resolved = path.resolve()
    cached = _TEMPLATE_CACHE.get(resolved)
    if cached is not None:
        return copy.deepcopy(cached)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise InvalidRequestError("Workflow template must be a JSON object")
    _TEMPLATE_CACHE[resolved] = copy.deepcopy(raw)
    return copy.deepcopy(raw)


def _next_node_ids(existing: dict[str, Any], count: int) -> list[str]:
    numeric = [int(key) for key in existing if str(key).isdigit()]
    start = max(numeric, default=0) + 1
    return [str(start + offset) for offset in range(count)]


def _strip_reference_image_inputs(text_encode: dict[str, Any]) -> None:
    inputs = text_encode.setdefault("inputs", {})
    for key in list(inputs):
        if key.startswith("images.image_"):
            del inputs[key]


def validate_generation_params(
    config: ImageGenerationConfig,
    *,
    resolution: int | None,
    steps: int | None,
    cfg: float | None,
    sampler_name: str | None,
    scheduler: str | None,
    denoise: float | None,
    input_image_count: int,
    allowed_samplers: frozenset[str] | None = None,
    allowed_schedulers: frozenset[str] | None = None,
) -> None:
    if input_image_count > config.max_input_images:
        raise TooManyImagesError(
            f"At most {config.max_input_images} input images are allowed",
            count=input_image_count,
            max_images=config.max_input_images,
        )

    if resolution is not None:
        res = resolution
    else:
        res = config.default_resolution
    if res < config.min_resolution or res > config.max_resolution:
        raise UnsupportedParameterError(
            "resolution out of allowed range",
            resolution=res,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
        )

    step_val = steps if steps is not None else config.default_steps
    if step_val < config.min_steps or step_val > config.max_steps:
        raise UnsupportedParameterError(
            "steps out of allowed range",
            steps=step_val,
            min_steps=config.min_steps,
            max_steps=config.max_steps,
        )

    cfg_val = cfg if cfg is not None else config.default_cfg
    if cfg_val < config.min_cfg or cfg_val > config.max_cfg:
        raise UnsupportedParameterError(
            "cfg out of allowed range",
            cfg=cfg_val,
            min_cfg=config.min_cfg,
            max_cfg=config.max_cfg,
        )

    denoise_val = denoise if denoise is not None else config.default_denoise
    if denoise_val < config.min_denoise or denoise_val > config.max_denoise:
        raise UnsupportedParameterError(
            "denoise out of allowed range",
            denoise=denoise_val,
            min_denoise=config.min_denoise,
            max_denoise=config.max_denoise,
        )

    sampler = sampler_name or config.default_sampler
    if allowed_samplers is not None and sampler not in allowed_samplers:
        raise UnsupportedParameterError(
            "unsupported sampler_name",
            sampler_name=sampler,
            allowed=sorted(allowed_samplers),
        )

    sched = scheduler or config.default_scheduler
    if allowed_schedulers is not None and sched not in allowed_schedulers:
        raise UnsupportedParameterError(
            "unsupported scheduler",
            scheduler=sched,
            allowed=sorted(allowed_schedulers),
        )


PROBE_NODE_CLASSES = frozenset(
    {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImage21",
        "KSampler",
        "VAEDecode",
        "SaveImage",
        "LoadImage",
    }
)


def validate_template_path(path: Path) -> None:
    load_workflow_template(path)


def validate_object_info(object_info: dict[str, Any]) -> None:
    if not isinstance(object_info, dict):
        raise InvalidRequestError("object_info must be a mapping")
    missing = sorted(PROBE_NODE_CLASSES - set(object_info.keys()))
    if missing:
        raise InvalidRequestError(
            "ComfyUI is missing required node classes",
            missing_nodes=missing,
        )


def _combo_options(field: Any) -> list[str]:
    if not isinstance(field, (list, tuple)) or not field:
        return []

    first = field[0]

    if isinstance(first, (list, tuple)):
        return [str(item) for item in first]

    return []


def extract_sampler_capabilities(object_info: dict[str, Any]) -> dict[str, list[str]]:
    """Read KSampler sampler/scheduler enums from ComfyUI ``/object_info``."""
    node = object_info.get("KSampler")

    if not isinstance(node, dict):
        return {}

    required = node.get("input", {})

    if not isinstance(required, dict):
        required = {}

    required_inputs = required.get("required", {})

    if not isinstance(required_inputs, dict):
        required_inputs = {}

    return {
        "sampler_name": _combo_options(required_inputs.get("sampler_name")),
        "scheduler": _combo_options(required_inputs.get("scheduler")),
    }


def build_workflow(
    *,
    template_path: Path,
    request: Any,
    request_id: str,
    reference_image_names: tuple[str, ...] = (),
    config: ImageGenerationConfig | None = None,
) -> dict[str, Any]:
    """Facade used by ``image_generation.ComfyUIBackend``."""
    cfg = config or load_image_generation_config()
    params = WorkflowBuildParams(
        request_id=request_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt or "",
        width=request.width,
        height=request.height,
        seed=request.seed,
        steps=request.steps,
        cfg=request.cfg,
        sampler_name=request.sampler,
        scheduler=request.scheduler,
        input_image_names=reference_image_names,
    )
    return build_image_workflow(params, cfg, template=load_workflow_template(template_path)).workflow


def build_image_workflow(
    params: WorkflowBuildParams,
    config: ImageGenerationConfig,
    *,
    template: dict[str, Any] | None = None,
    allowed_samplers: frozenset[str] | None = None,
    allowed_schedulers: frozenset[str] | None = None,
) -> WorkflowBuildResult:
    """Deep-copy the template and apply generation parameters for one request."""
    image_inputs = resolved_workflow_image_inputs(params)
    square_resolution = resolve_square_resolution(
        config,
        width=params.width,
        height=params.height,
        resolution=params.resolution,
    )
    validate_generation_params(
        config,
        resolution=square_resolution,
        steps=params.steps,
        cfg=params.cfg,
        sampler_name=params.sampler_name,
        scheduler=params.scheduler,
        denoise=params.denoise,
        input_image_count=len(image_inputs),
        allowed_samplers=allowed_samplers,
        allowed_schedulers=allowed_schedulers,
    )

    workflow = copy.deepcopy(
        template
        if template is not None
        else load_workflow_template(config.workflow_template_path),
    )

    seed_used = params.seed
    if seed_used is None:
        seed_used = secrets.randbelow(2**63 - 1)

    text_encode = workflow[TEXT_ENCODE_NODE_ID]
    _strip_reference_image_inputs(text_encode)
    text_encode["inputs"]["prompt"] = augment_prompt_for_image_roles(
        params.prompt,
        image_inputs,
    )
    text_encode["inputs"]["negative_prompt"] = params.negative_prompt
    text_encode["inputs"]["resolution"] = square_resolution

    ksampler = workflow[KSAMPLER_NODE_ID]
    ksampler["inputs"]["seed"] = seed_used
    ksampler["inputs"]["steps"] = (
        params.steps if params.steps is not None else config.default_steps
    )
    ksampler["inputs"]["cfg"] = (
        params.cfg if params.cfg is not None else config.default_cfg
    )
    ksampler["inputs"]["sampler_name"] = (
        params.sampler_name
        if params.sampler_name is not None
        else config.default_sampler
    )
    ksampler["inputs"]["scheduler"] = (
        params.scheduler
        if params.scheduler is not None
        else config.default_scheduler
    )
    ksampler["inputs"]["denoise"] = (
        params.denoise if params.denoise is not None else config.default_denoise
    )

    save_image = workflow[SAVE_IMAGE_NODE_ID]
    save_image["inputs"]["filename_prefix"] = f"mcp/{params.request_id}/result"

    if image_inputs:
        load_ids = _next_node_ids(workflow, len(image_inputs))
        for index, (node_id, slot) in enumerate(zip(load_ids, image_inputs), start=1):
            workflow[node_id] = {
                "inputs": {"image": slot.filename},
                "class_type": "LoadImage",
                "_meta": {"title": f"LoadImage ({slot.role})"},
            }
            text_encode["inputs"][f"images.image_{index}"] = [node_id, 0]

    return WorkflowBuildResult(workflow=workflow, seed_used=seed_used)


def choose_random_seed() -> int:
    """Return a non-negative seed suitable for KSampler."""
    return secrets.randbelow(2**63 - 1)


def resolve_comfy_output_path(
    output_root: Path,
    filename: str,
    subfolder: str = "",
    folder_type: str = "output",
) -> Path:
    """Resolve a ComfyUI output file under ``output_root`` (anti-traversal)."""
    if folder_type != "output":
        raise OutputInvalidError(
            "Only output folder type is supported for reads",
            folder_type=folder_type,
        )
    if not filename or filename != Path(filename).name:
        raise OutputInvalidError("Invalid output filename", filename=filename)
    if ".." in Path(subfolder).parts or Path(subfolder).is_absolute():
        raise OutputInvalidError("Invalid output subfolder", subfolder=subfolder)

    root = output_root.resolve()
    candidate = (root / subfolder / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise OutputInvalidError(
            "Output path escapes configured root",
            filename=filename,
            subfolder=subfolder,
        ) from None

    if candidate.is_symlink():
        real = candidate.resolve()
        try:
            real.relative_to(root)
        except ValueError:
            raise OutputInvalidError(
                "Output symlink escapes configured root",
                filename=filename,
                subfolder=subfolder,
            ) from None
        return real

    return candidate
