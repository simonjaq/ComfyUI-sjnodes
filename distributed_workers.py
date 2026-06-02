from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import requests


class WorkerDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerDescriptor:
    port: int
    url: str
    name: str
    gpu_name: str
    gpu_uuid: str
    healthy: bool
    state: str
    cuda_device: Optional[int]
    queue_running: int = 0
    queue_pending: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_requested_ports(requested_ports: Any) -> List[int]:
    if requested_ports is None:
        return [8188, 8190]
    if isinstance(requested_ports, int):
        return [requested_ports]
    if isinstance(requested_ports, str):
        raw = [part.strip() for part in requested_ports.split(",") if part.strip()]
        return [int(part) for part in raw]
    if isinstance(requested_ports, Iterable):
        return [int(part) for part in requested_ports]
    raise WorkerDiscoveryError(f"unsupported worker_ports value: {requested_ports!r}")


def fetch_dashboard_status(status_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    response = requests.get(status_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise WorkerDiscoveryError("dashboard status payload must be a JSON object")
    return payload


def fetch_queue_state(worker_url: str, timeout: float = 5.0) -> Dict[str, int]:
    response = requests.get(f"{worker_url.rstrip('/')}/queue", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    running = payload.get("queue_running", [])
    pending = payload.get("queue_pending", [])
    return {
        "queue_running": len(running) if isinstance(running, list) else int(bool(running)),
        "queue_pending": len(pending) if isinstance(pending, list) else int(bool(pending)),
    }


def discover_workers(
    status_url: str,
    requested_ports: Any = None,
    require_gpu_substring: str = "5090",
    timeout: float = 5.0,
) -> List[WorkerDescriptor]:
    status_payload = fetch_dashboard_status(status_url, timeout=timeout)
    ports = normalize_requested_ports(requested_ports)
    wanted_ports = set(ports)

    descriptors: List[WorkerDescriptor] = []
    instances = status_payload.get("instances", [])
    if not isinstance(instances, list):
        raise WorkerDiscoveryError("dashboard status missing instances list")

    for instance in instances:
        if not isinstance(instance, dict):
            continue
        port = instance.get("port")
        if port not in wanted_ports:
            continue
        gpu_name = (((instance.get("gpu") or {}).get("host") or {}).get("name") or "").strip()
        if require_gpu_substring and require_gpu_substring.lower() not in gpu_name.lower():
            continue
        worker_url = f"http://127.0.0.1:{port}"
        queue_state = fetch_queue_state(worker_url, timeout=timeout)
        descriptors.append(
            WorkerDescriptor(
                port=int(port),
                url=worker_url,
                name=str(instance.get("name", f"Comfy {port}")),
                gpu_name=gpu_name,
                gpu_uuid=str((((instance.get("gpu") or {}).get("host") or {}).get("uuid") or "")),
                healthy=bool(instance.get("healthy")),
                state=str(instance.get("state", "unknown")),
                cuda_device=instance.get("cuda_device"),
                queue_running=queue_state["queue_running"],
                queue_pending=queue_state["queue_pending"],
            )
        )

    found_ports = {worker.port for worker in descriptors}
    missing = sorted(wanted_ports - found_ports)
    if missing:
        raise WorkerDiscoveryError(
            f"failed to validate requested worker ports {missing}; expected healthy 5090-backed instances from dashboard {status_url}"
        )

    unhealthy = [worker.port for worker in descriptors if not worker.healthy or worker.state != "healthy"]
    if unhealthy:
        raise WorkerDiscoveryError(f"requested workers are not healthy: {unhealthy}")

    return sorted(descriptors, key=lambda item: item.port)


def choose_next_worker(active_by_port: Mapping[int, Any], workers: Sequence[WorkerDescriptor]) -> Optional[WorkerDescriptor]:
    for worker in workers:
        if active_by_port.get(worker.port) is None:
            return worker
    return None


def snapshot_workers_json(workers: Sequence[WorkerDescriptor]) -> str:
    return json.dumps([worker.to_dict() for worker in workers], indent=2, sort_keys=True)
