"""Pure and live-CUDA preflight for the non-simulator sweep environment."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alexdoor_xas.cluster_pilot.preflight import (
    REQUIRED_IMPORTS,
    ClusterPreflightError,
    dependency_inventory,
    isaac_module_inventory,
    probe_cuda_device,
)

from .config import SweepConfig
from .transfer import verify_sweep_transfer_manifest


def run_sweep_preflight(
    *,
    repo_root: str | Path,
    config: SweepConfig,
    manifest: dict[str, Any],
    scratch_output: str | Path,
    source_state: dict[str, Any] | None = None,
    dependency_probe: Callable[[], dict[str, str]] | None = None,
    module_probe: Callable[[], dict[str, str]] = isaac_module_inventory,
    checkpoint_probe: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    dependencies = (dependency_probe or sweep_dependency_inventory)()
    python_version = dependencies.get("python", "")
    if not python_version.startswith(f"{config.environment.python_major_minor}."):
        raise ClusterPreflightError(
            f"Python must be {config.environment.python_major_minor}.x, got {python_version!r}"
        )
    missing_dependencies = sorted({"python", *REQUIRED_IMPORTS, "ruff"} - set(dependencies))
    if missing_dependencies:
        raise ClusterPreflightError(
            f"dependency inventory is incomplete: {missing_dependencies}"
        )
    expected_versions = {
        "numpy": config.environment.numpy_version,
        "torch": config.environment.torch_version,
        "torch_cuda": config.environment.torch_cuda_version,
    }
    for name, expected in expected_versions.items():
        if dependencies.get(name) != expected:
            raise ClusterPreflightError(
                f"{name} runtime must be {expected}, got {dependencies.get(name)!r}"
            )
    modules = module_probe()
    if modules:
        raise ClusterPreflightError(
            "simulator namespaces must be absent on the cluster: " + ", ".join(sorted(modules))
        )
    failures = verify_sweep_transfer_manifest(
        manifest,
        repo_root,
        config,
        source_state=source_state,
        require_tracked=source_state is None,
    )
    if failures:
        raise ClusterPreflightError("sweep transfer verification failed: " + "; ".join(failures))
    if config.training.wandb_mode != "offline":
        raise ClusterPreflightError("full sweep W&B mode must be explicitly offline")
    scratch = Path(scratch_output).resolve()
    if not scratch.is_dir():
        raise ClusterPreflightError(f"scratch output directory does not exist: {scratch}")
    try:
        with tempfile.NamedTemporaryFile(
            dir=scratch, prefix=".write-probe-", delete=True
        ) as handle:
            handle.write(b"scale-sweep-write-probe")
            handle.flush()
    except OSError as error:
        raise ClusterPreflightError(
            f"scratch output is not writable: {scratch}: {error}"
        ) from error
    checkpoint = scratch / ".scale-sweep-checkpoint-probe.pt"
    checkpoint.unlink(missing_ok=True)
    try:
        (checkpoint_probe or _torch_checkpoint_probe)(checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ClusterPreflightError("checkpoint round-trip probe produced no file")
    finally:
        checkpoint.unlink(missing_ok=True)
    return {
        "schema": "alexdoor_xas.cluster_sweep_preflight_report.v1",
        "status": "PASS",
        "python": python_version,
        "dependencies": dict(sorted(dependencies.items())),
        "simulator_modules": {},
        "source_git_commit": manifest["source_git"]["commit"],
        "manifest_schema": manifest["schema"],
        "master_version": config.dataset.master_version,
        "view_ids": [view.view_id for view in config.views],
        "cell_count": len(config.cells),
        "scratch_output": str(scratch),
        "checkpoint_round_trip": "PASS",
        "wandb_mode": config.training.wandb_mode,
        "cuda_probe": "NOT_RUN",
    }


def sweep_dependency_inventory() -> dict[str, str]:
    """Add the Torch CUDA build to the proven non-Isaac dependency inventory."""
    inventory = dependency_inventory()
    import torch

    inventory["torch_cuda"] = str(torch.version.cuda or "")
    return inventory


def _torch_checkpoint_probe(path: Path) -> None:
    import torch

    expected = torch.arange(8, dtype=torch.float32)
    torch.save({"probe": expected}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(loaded["probe"], expected):
        raise ClusterPreflightError("checkpoint round-trip contents differ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ClusterPreflightError",
    "atomic_json",
    "probe_cuda_device",
    "run_sweep_preflight",
    "sweep_dependency_inventory",
]
