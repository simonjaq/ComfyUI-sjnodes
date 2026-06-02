from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import requests
import torch
from PIL import Image

from .distributed_workflows import (
    WorkflowValidationError,
    extract_output_artifacts,
    load_api_workflow,
    load_mapping_config,
    patch_workflow_for_image_job,
    validate_mapping_against_workflow,
)
from .distributed_workers import (
    WorkerDescriptor,
    choose_next_worker,
    discover_workers,
)


class DistributedBatchRuntimeError(RuntimeError):
    pass


@dataclass
class ShotJob:
    shot_index: int
    prompt: str
    ref_paths: List[str]
    output_prefix: str


@dataclass
class ShotResult:
    shot_index: int
    status: str
    worker_port: Optional[int]
    prompt_id: Optional[str]
    output_prefix: str
    image_paths: List[str]
    ref_paths: List[str]
    prompt: str
    error: Optional[str] = None
    history_summary: Optional[Dict[str, Any]] = None
    patch_report: Optional[Dict[str, Any]] = None


@dataclass
class ActiveJob:
    job: ShotJob
    worker: WorkerDescriptor
    prompt_id: str
    output_prefix: str
    patch_report: Dict[str, Any]
    started_at: float


class ComfyWorkerClient:
    def __init__(self, worker: WorkerDescriptor, timeout: float = 10.0):
        self.worker = worker
        self.timeout = timeout

    def submit_prompt(self, prompt_workflow: Mapping[str, Any], client_id: str) -> str:
        response = requests.post(
            f"{self.worker.url}/prompt",
            json={"prompt": prompt_workflow, "client_id": client_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise DistributedBatchRuntimeError(
                f"worker {self.worker.port} did not return prompt_id: {payload}"
            )
        return str(prompt_id)

    def fetch_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        response = requests.get(f"{self.worker.url}/history/{prompt_id}", timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            if prompt_id in payload and isinstance(payload[prompt_id], dict):
                return payload[prompt_id]
            if payload.get("prompt") or payload.get("outputs") or payload.get("status"):
                return payload
        return None


class SJDistributedImageBatch:
    CATEGORY = "sjnodes/Distributed"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("image_paths", "metadata_json", "shot_count")
    OUTPUT_IS_LIST = (True, False, False)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Prompt list or newline-separated prompts."}),
                "worker_workflow_path": ("STRING", {"default": "", "multiline": False}),
                "mapping_json": ("STRING", {"default": json.dumps(cls.example_mapping(), indent=2), "multiline": True}),
                "submission_mode": (["validate_only", "submit"], {"default": "validate_only"}),
                "worker_ports": ("STRING", {"default": "8188,8190"}),
                "dashboard_status_url": ("STRING", {"default": "http://127.0.0.1:8210/api/status"}),
                "output_root": ("STRING", {"default": "./output/sj_distributed_image_batch"}),
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
            },
        }

    @staticmethod
    def example_mapping() -> Dict[str, Any]:
        return {
            "prompt": {"node_id": "12", "input_name": "text"},
            "refs": [
                {"node_id": "45", "input_name": "image_path"},
                {"node_id": "46", "input_name": "image_path"},
                {"node_id": "47", "input_name": "image_path"},
            ],
            "output": {"node_id": "99", "filename_prefix_input": "filename_prefix"},
        }

    def run(
        self,
        prompts,
        worker_workflow_path,
        mapping_json,
        submission_mode,
        worker_ports,
        dashboard_status_url,
        output_root,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
    ):
        prompts_list = self._normalize_prompts(prompts)
        config = {
            "worker_workflow_path": self._first_scalar(worker_workflow_path),
            "mapping_json": self._first_scalar(mapping_json),
            "submission_mode": self._first_scalar(submission_mode),
            "worker_ports": self._first_scalar(worker_ports),
            "dashboard_status_url": self._first_scalar(dashboard_status_url),
            "output_root": self._first_scalar(output_root),
        }

        if not prompts_list:
            raise DistributedBatchRuntimeError("SJ Distributed Image Batch requires at least one prompt")

        output_root_abs = self._resolve_output_root(config["output_root"])
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        run_root = os.path.join(output_root_abs, run_id)
        os.makedirs(run_root, exist_ok=True)

        workflow = load_api_workflow(config["worker_workflow_path"])
        mapping = load_mapping_config(config["mapping_json"])
        mapping_errors = validate_mapping_against_workflow(workflow, mapping)
        if mapping_errors:
            raise WorkflowValidationError("; ".join(mapping_errors))

        workers = discover_workers(
            status_url=config["dashboard_status_url"],
            requested_ports=config["worker_ports"],
        )

        ref_slots = [ref_image_1, ref_image_2, ref_image_3]
        ref_paths_by_shot = self._materialize_ref_paths(ref_slots, len(prompts_list), run_root)
        jobs = self._build_jobs(prompts_list, ref_paths_by_shot, run_id)

        metadata: Dict[str, Any] = {
            "status": "validated" if config["submission_mode"] == "validate_only" else "submitted",
            "run_id": run_id,
            "run_root": run_root,
            "workflow_path": config["worker_workflow_path"],
            "worker_ports": [worker.port for worker in workers],
            "workers": [asdict(worker) for worker in workers],
            "shot_count": len(jobs),
            "mode": config["submission_mode"],
            "shots": [],
        }

        planned_paths: List[str] = []
        for job in jobs:
            patched = patch_workflow_for_image_job(workflow, mapping, job.prompt, job.ref_paths, job.output_prefix)
            planned_paths.append(job.output_prefix)
            metadata["shots"].append({
                "shot_index": job.shot_index,
                "output_prefix": job.output_prefix,
                "prompt_preview": job.prompt[:160],
                "ref_paths": job.ref_paths,
                "patch_report": patched.patch_report,
            })

        if config["submission_mode"] == "validate_only":
            return (planned_paths, json.dumps(metadata, indent=2, sort_keys=True), len(jobs))

        root_by_type = self._build_root_by_type()
        results = self._run_scheduler(jobs, workers, workflow, mapping, root_by_type)
        ordered_results = self._restore_order(results, len(jobs))
        image_paths = [result.image_paths[0] if result.image_paths else "" for result in ordered_results]
        metadata["shots"] = [asdict(result) for result in ordered_results]
        metadata["status"] = self._overall_status(ordered_results)
        return (image_paths, json.dumps(metadata, indent=2, sort_keys=True), len(jobs))

    @staticmethod
    def _first_scalar(value: Any) -> Any:
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _resolve_output_root(path: str) -> str:
        path = (path or "").strip().strip('"')
        if not path:
            path = "./output/sj_distributed_image_batch"
        if not os.path.isabs(path):
            import folder_paths
            path = os.path.join(folder_paths.base_path, path)
        return os.path.abspath(path)

    @staticmethod
    def _normalize_prompts(raw_prompts: Any) -> List[str]:
        if isinstance(raw_prompts, list):
            merged: List[str] = []
            for item in raw_prompts:
                merged.extend(SJDistributedImageBatch._normalize_prompts(item))
            return merged
        if isinstance(raw_prompts, str):
            stripped = raw_prompts.strip()
            if not stripped:
                return []
            try:
                payload = json.loads(stripped)
                if isinstance(payload, list):
                    return [str(entry) for entry in payload if str(entry).strip()]
            except json.JSONDecodeError:
                pass
            return [line.strip() for line in stripped.splitlines() if line.strip()]
        return [str(raw_prompts)]

    @staticmethod
    def _flatten_image_input(raw_value: Any) -> List[torch.Tensor]:
        images: List[torch.Tensor] = []
        if raw_value is None:
            return images
        if isinstance(raw_value, list):
            for item in raw_value:
                images.extend(SJDistributedImageBatch._flatten_image_input(item))
            return images
        if isinstance(raw_value, torch.Tensor):
            if raw_value.ndim == 4:
                for idx in range(raw_value.shape[0]):
                    images.append(raw_value[idx:idx+1].detach().cpu())
                return images
            if raw_value.ndim == 3:
                images.append(raw_value.unsqueeze(0).detach().cpu())
                return images
        raise DistributedBatchRuntimeError(f"unsupported IMAGE input shape/type for distributed batch: {type(raw_value)!r}")

    def _materialize_ref_paths(self, ref_slots: Sequence[Any], shot_count: int, run_root: str) -> List[List[str]]:
        refs_root = os.path.join(run_root, "refs")
        os.makedirs(refs_root, exist_ok=True)
        slot_images = [self._flatten_image_input(slot) for slot in ref_slots]
        for slot_idx, images in enumerate(slot_images, start=1):
            if images and len(images) not in (1, shot_count):
                raise DistributedBatchRuntimeError(
                    f"ref_image_{slot_idx} supplied {len(images)} image(s), expected either 1 or {shot_count}"
                )

        refs_by_shot: List[List[str]] = []
        for shot_index in range(shot_count):
            shot_refs: List[str] = []
            for slot_idx, images in enumerate(slot_images, start=1):
                if not images:
                    continue
                image = images[0] if len(images) == 1 else images[shot_index]
                slot_dir = os.path.join(refs_root, f"slot_{slot_idx}")
                os.makedirs(slot_dir, exist_ok=True)
                path = os.path.join(slot_dir, f"shot_{shot_index:04d}.png")
                self._save_tensor_image(image, path)
                shot_refs.append(path)
            refs_by_shot.append(shot_refs)
        return refs_by_shot

    @staticmethod
    def _save_tensor_image(tensor: torch.Tensor, path: str) -> None:
        data = tensor.detach().cpu()
        if data.ndim == 4:
            data = data[0]
        array = data.numpy()
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(path)

    @staticmethod
    def _build_jobs(prompts: Sequence[str], ref_paths_by_shot: Sequence[Sequence[str]], run_id: str) -> List[ShotJob]:
        jobs: List[ShotJob] = []
        for idx, prompt in enumerate(prompts):
            output_prefix = f"sj_distributed/{run_id}/shot_{idx:04d}"
            jobs.append(
                ShotJob(
                    shot_index=idx,
                    prompt=prompt,
                    ref_paths=list(ref_paths_by_shot[idx]) if idx < len(ref_paths_by_shot) else [],
                    output_prefix=output_prefix,
                )
            )
        return jobs

    @staticmethod
    def _build_root_by_type() -> Dict[str, str]:
        import folder_paths
        return {
            "output": folder_paths.get_output_directory(),
            "temp": folder_paths.get_temp_directory(),
            "input": folder_paths.get_input_directory(),
        }

    def _run_scheduler(
        self,
        jobs: Sequence[ShotJob],
        workers: Sequence[WorkerDescriptor],
        workflow_template: Mapping[str, Any],
        mapping,
        root_by_type: Mapping[str, str],
        poll_interval: float = 1.5,
    ) -> List[ShotResult]:
        active_by_port: Dict[int, Optional[ActiveJob]] = {worker.port: None for worker in workers}
        pending_jobs = list(jobs)
        completed: List[ShotResult] = []
        clients = {worker.port: ComfyWorkerClient(worker) for worker in workers}
        client_id = f"sj-distributed-image-batch-{uuid.uuid4().hex}"

        while pending_jobs or any(active_by_port.values()):
            while pending_jobs:
                worker = choose_next_worker(active_by_port, workers)
                if worker is None:
                    break
                job = pending_jobs.pop(0)
                patched = patch_workflow_for_image_job(
                    workflow_template,
                    mapping,
                    job.prompt,
                    job.ref_paths,
                    job.output_prefix,
                )
                try:
                    prompt_id = clients[worker.port].submit_prompt(patched.workflow, client_id=client_id)
                    active_by_port[worker.port] = ActiveJob(
                        job=job,
                        worker=worker,
                        prompt_id=prompt_id,
                        output_prefix=job.output_prefix,
                        patch_report=patched.patch_report,
                        started_at=time.time(),
                    )
                except Exception as exc:
                    completed.append(
                        ShotResult(
                            shot_index=job.shot_index,
                            status="submit_failed",
                            worker_port=worker.port,
                            prompt_id=None,
                            output_prefix=job.output_prefix,
                            image_paths=[],
                            ref_paths=job.ref_paths,
                            prompt=job.prompt,
                            error=str(exc),
                            patch_report=patched.patch_report,
                        )
                    )

            for worker in workers:
                active = active_by_port.get(worker.port)
                if active is None:
                    continue
                try:
                    history = clients[worker.port].fetch_history(active.prompt_id)
                except Exception as exc:
                    completed.append(
                        ShotResult(
                            shot_index=active.job.shot_index,
                            status="history_error",
                            worker_port=worker.port,
                            prompt_id=active.prompt_id,
                            output_prefix=active.output_prefix,
                            image_paths=[],
                            ref_paths=active.job.ref_paths,
                            prompt=active.job.prompt,
                            error=str(exc),
                            patch_report=active.patch_report,
                        )
                    )
                    active_by_port[worker.port] = None
                    continue

                if history is None:
                    continue

                artifacts = extract_output_artifacts(history, active.patch_report["output"]["node_id"], root_by_type)
                status_block = history.get("status") if isinstance(history, dict) else {}
                status_str = status_block.get("status_str", "success") if isinstance(status_block, dict) else "success"
                result_status = "success" if artifacts else f"{status_str}_no_output"
                completed.append(
                    ShotResult(
                        shot_index=active.job.shot_index,
                        status=result_status,
                        worker_port=worker.port,
                        prompt_id=active.prompt_id,
                        output_prefix=active.output_prefix,
                        image_paths=[artifact.abs_path for artifact in artifacts],
                        ref_paths=active.job.ref_paths,
                        prompt=active.job.prompt,
                        error=None if artifacts else "worker completed without output images on mapped node",
                        history_summary={
                            "status": status_block,
                            "outputs": history.get("outputs", {}),
                        },
                        patch_report=active.patch_report,
                    )
                )
                active_by_port[worker.port] = None

            if pending_jobs or any(active_by_port.values()):
                time.sleep(poll_interval)

        return completed

    @staticmethod
    def _restore_order(results: Sequence[ShotResult], shot_count: int) -> List[ShotResult]:
        indexed = {result.shot_index: result for result in results}
        ordered: List[ShotResult] = []
        for idx in range(shot_count):
            ordered.append(
                indexed.get(
                    idx,
                    ShotResult(
                        shot_index=idx,
                        status="missing",
                        worker_port=None,
                        prompt_id=None,
                        output_prefix="",
                        image_paths=[],
                        ref_paths=[],
                        prompt="",
                        error="missing result",
                    ),
                )
            )
        return ordered

    @staticmethod
    def _overall_status(results: Sequence[ShotResult]) -> str:
        statuses = {result.status for result in results}
        if statuses == {"success"}:
            return "success"
        if "success" in statuses:
            return "partial_failure"
        return "failed"
