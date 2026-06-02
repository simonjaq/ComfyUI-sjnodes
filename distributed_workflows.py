import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_WORKFLOW_ALLOWLIST = (
    "/home/simon/coding/ComfyUI/custom_nodes/ComfyUI-sjnodes/workflows",
    "/home/simon/coding/comfy-dashboard/workflows",
    "/home/simon/coding/comfy-dashboard/prepared_workflows",
    "/home/simon/coding/human-in-the-loop/endless_story/comfyui_workflows",
)


class WorkflowValidationError(ValueError):
    pass


@dataclass(frozen=True)
class InputBinding:
    node_id: str
    input_name: str


@dataclass(frozen=True)
class OutputBinding:
    node_id: str
    filename_prefix_input: str


@dataclass(frozen=True)
class WorkflowMapping:
    prompt: InputBinding
    refs: List[InputBinding]
    output: OutputBinding


@dataclass(frozen=True)
class PatchedWorkflow:
    workflow: Dict[str, Any]
    output_node_id: str
    output_prefix: str
    patch_report: Dict[str, Any]


@dataclass(frozen=True)
class WorkflowArtifact:
    filename: str
    subfolder: str
    storage_type: str
    abs_path: str


def normalize_allowlisted_roots(extra_roots: Optional[Iterable[str]] = None) -> List[str]:
    roots = list(DEFAULT_WORKFLOW_ALLOWLIST)
    for root in extra_roots or []:
        if root and root not in roots:
            roots.append(root)
    return [os.path.abspath(os.path.expanduser(root)) for root in roots]


def is_path_allowlisted(path: str, allowlisted_roots: Sequence[str]) -> bool:
    candidate = os.path.abspath(os.path.expanduser(path))
    for root in allowlisted_roots:
        try:
            common = os.path.commonpath([candidate, root])
        except ValueError:
            continue
        if common == root:
            return True
    return False


def load_api_workflow(workflow_path: str, allowlisted_roots: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if not workflow_path:
        raise WorkflowValidationError("worker workflow path is required")

    roots = normalize_allowlisted_roots(allowlisted_roots)
    resolved = os.path.abspath(os.path.expanduser(workflow_path.strip().strip('"')))
    if not is_path_allowlisted(resolved, roots):
        raise WorkflowValidationError(
            f"workflow path '{resolved}' is outside allowlisted roots: {roots}"
        )
    if not os.path.isfile(resolved):
        raise WorkflowValidationError(f"workflow path does not exist: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise WorkflowValidationError("API workflow JSON must be an object keyed by node id")

    # Reject UI workflow wrappers early; v1 expects raw API prompt JSON.
    if "nodes" in payload or "links" in payload:
        raise WorkflowValidationError(
            "workflow JSON looks like a UI graph export; v1 requires API-format prompt JSON"
        )

    for node_id, node_data in payload.items():
        if not isinstance(node_id, str) or not isinstance(node_data, dict):
            raise WorkflowValidationError("workflow JSON must be a dict of node_id -> node object")
        if "inputs" not in node_data or not isinstance(node_data["inputs"], dict):
            raise WorkflowValidationError(f"workflow node '{node_id}' is missing an inputs object")

    return payload


def load_mapping_config(mapping_json: str) -> WorkflowMapping:
    if not mapping_json or not mapping_json.strip():
        raise WorkflowValidationError("mapping JSON is required")
    try:
        payload = json.loads(mapping_json)
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(f"mapping JSON is invalid: {exc}") from exc

    if not isinstance(payload, dict):
        raise WorkflowValidationError("mapping JSON must be an object")

    prompt = _parse_input_binding(payload.get("prompt"), field_name="prompt")
    refs_payload = payload.get("refs", [])
    if refs_payload is None:
        refs_payload = []
    if not isinstance(refs_payload, list):
        raise WorkflowValidationError("mapping.refs must be a list")
    if len(refs_payload) > 3:
        raise WorkflowValidationError("mapping.refs may contain at most 3 entries")
    refs = [_parse_input_binding(item, field_name=f"refs[{idx}]") for idx, item in enumerate(refs_payload)]

    output_payload = payload.get("output")
    if not isinstance(output_payload, dict):
        raise WorkflowValidationError("mapping.output must be an object")
    node_id = str(output_payload.get("node_id", "")).strip()
    filename_prefix_input = str(output_payload.get("filename_prefix_input", "")).strip()
    if not node_id or not filename_prefix_input:
        raise WorkflowValidationError(
            "mapping.output requires non-empty node_id and filename_prefix_input"
        )

    return WorkflowMapping(
        prompt=prompt,
        refs=refs,
        output=OutputBinding(node_id=node_id, filename_prefix_input=filename_prefix_input),
    )


def _parse_input_binding(payload: Any, field_name: str) -> InputBinding:
    if not isinstance(payload, dict):
        raise WorkflowValidationError(f"mapping.{field_name} must be an object")
    node_id = str(payload.get("node_id", "")).strip()
    input_name = str(payload.get("input_name", "")).strip()
    if not node_id or not input_name:
        raise WorkflowValidationError(f"mapping.{field_name} requires node_id and input_name")
    return InputBinding(node_id=node_id, input_name=input_name)


def validate_mapping_against_workflow(workflow: Mapping[str, Any], mapping: WorkflowMapping) -> List[str]:
    errors: List[str] = []
    _validate_binding(workflow, mapping.prompt, "prompt", errors)
    for idx, ref_binding in enumerate(mapping.refs):
        _validate_binding(workflow, ref_binding, f"refs[{idx}]", errors)

    output_node = workflow.get(mapping.output.node_id)
    if output_node is None:
        errors.append(f"output node '{mapping.output.node_id}' is missing from workflow")
    else:
        inputs = output_node.get("inputs")
        if not isinstance(inputs, dict):
            errors.append(f"output node '{mapping.output.node_id}' is missing inputs")
        elif mapping.output.filename_prefix_input not in inputs:
            errors.append(
                f"output node '{mapping.output.node_id}' is missing input '{mapping.output.filename_prefix_input}'"
            )

    return errors


def _validate_binding(workflow: Mapping[str, Any], binding: InputBinding, name: str, errors: List[str]) -> None:
    node = workflow.get(binding.node_id)
    if node is None:
        errors.append(f"{name} node '{binding.node_id}' is missing from workflow")
        return
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        errors.append(f"{name} node '{binding.node_id}' is missing inputs")
        return
    if binding.input_name not in inputs:
        errors.append(
            f"{name} node '{binding.node_id}' is missing input '{binding.input_name}'"
        )


def patch_workflow_for_image_job(
    workflow: Mapping[str, Any],
    mapping: WorkflowMapping,
    prompt_text: str,
    ref_paths: Sequence[str],
    output_prefix: str,
) -> PatchedWorkflow:
    patched = copy.deepcopy(dict(workflow))
    patched[mapping.prompt.node_id]["inputs"][mapping.prompt.input_name] = prompt_text

    ref_report: List[Dict[str, Any]] = []
    for idx, binding in enumerate(mapping.refs):
        value = ref_paths[idx] if idx < len(ref_paths) else ""
        patched[binding.node_id]["inputs"][binding.input_name] = value
        ref_report.append({
            "slot": idx + 1,
            "node_id": binding.node_id,
            "input_name": binding.input_name,
            "value": value,
        })

    patched[mapping.output.node_id]["inputs"][mapping.output.filename_prefix_input] = output_prefix
    report = {
        "prompt": {
            "node_id": mapping.prompt.node_id,
            "input_name": mapping.prompt.input_name,
            "length": len(prompt_text),
        },
        "refs": ref_report,
        "output": {
            "node_id": mapping.output.node_id,
            "filename_prefix_input": mapping.output.filename_prefix_input,
            "value": output_prefix,
        },
    }
    return PatchedWorkflow(
        workflow=patched,
        output_node_id=mapping.output.node_id,
        output_prefix=output_prefix,
        patch_report=report,
    )


def extract_output_artifacts(
    history_payload: Mapping[str, Any],
    output_node_id: str,
    root_by_type: Mapping[str, str],
) -> List[WorkflowArtifact]:
    outputs = history_payload.get("outputs")
    if not isinstance(outputs, dict):
        return []
    node_outputs = outputs.get(output_node_id)
    if not isinstance(node_outputs, dict):
        return []

    images = node_outputs.get("images", [])
    artifacts: List[WorkflowArtifact] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "")).strip()
        subfolder = str(item.get("subfolder", "")).strip()
        storage_type = str(item.get("type", "output")).strip() or "output"
        root = root_by_type.get(storage_type)
        if not filename or not root:
            continue
        abs_path = os.path.join(root, subfolder, filename) if subfolder else os.path.join(root, filename)
        artifacts.append(
            WorkflowArtifact(
                filename=filename,
                subfolder=subfolder,
                storage_type=storage_type,
                abs_path=os.path.abspath(abs_path),
            )
        )
    return artifacts
