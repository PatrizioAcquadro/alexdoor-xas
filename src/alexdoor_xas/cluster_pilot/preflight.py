"""Pure and live-device preflight checks for the non-Isaac Gilbreth pilot."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .transfer import verify_transfer_manifest

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "diffusers": "diffusers",
    "wandb": "wandb",
    "pytest": "pytest",
    "torch": "torch",
}
ISAAC_TOP_LEVEL_MODULES = ("isaacsim", "isaaclab", "omni")


class ClusterPreflightError(RuntimeError):
    """Raised when a Gilbreth compatibility prerequisite is not proven."""


def dependency_inventory() -> dict[str, str]:
    """Import required runtime packages and return their resolved versions."""
    inventory = {"python": platform.python_version()}
    for module_name, distribution_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
            inventory[module_name] = importlib.metadata.version(distribution_name)
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise ClusterPreflightError(
                f"required dependency is unavailable: {distribution_name}: {error}"
            ) from error
    ruff = shutil.which("ruff")
    if ruff is None:
        raise ClusterPreflightError("required dependency is unavailable: ruff executable")
    result = subprocess.run(
        [ruff, "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ClusterPreflightError(f"ruff --version failed: {result.stderr.strip()}")
    inventory["ruff"] = result.stdout.strip()
    return inventory


def isaac_module_inventory() -> dict[str, str]:
    """Return installed/imported Isaac namespaces without importing them."""
    found: dict[str, str] = {}
    for name in ISAAC_TOP_LEVEL_MODULES:
        if name in sys.modules:
            found[name] = "IMPORTED"
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None:
            found[name] = str(spec.origin or "AVAILABLE")
    return found


def run_pure_preflight(
    *,
    repo_root: str | Path,
    config: PilotConfig,
    manifest: dict[str, Any],
    scratch_output: str | Path,
    source_state: dict[str, Any] | None = None,
    dependency_probe: Callable[[], dict[str, str]] = dependency_inventory,
    isaac_probe: Callable[[], dict[str, str]] = isaac_module_inventory,
    checkpoint_probe: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Run all non-device checks; CUDA is deliberately a separate live probe."""
    dependencies = dependency_probe()
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
    isaac_modules = isaac_probe()
    if isaac_modules:
        raise ClusterPreflightError(
            "Isaac modules must be absent and unimported on Gilbreth: "
            + ", ".join(sorted(isaac_modules))
        )
    failures = verify_transfer_manifest(
        manifest,
        repo_root,
        config,
        source_state=source_state,
        require_tracked=source_state is None,
    )
    if failures:
        raise ClusterPreflightError("pilot transfer verification failed: " + "; ".join(failures))
    if config.training.wandb_mode not in {"disabled", "offline", "online"}:
        raise ClusterPreflightError("W&B mode must be explicit")
    if config.training.wandb_mode != "offline":
        raise ClusterPreflightError(
            "the compatibility pilot must default explicitly to W&B offline"
        )

    scratch = Path(scratch_output).resolve()
    if not scratch.is_dir():
        raise ClusterPreflightError(f"scratch output directory does not exist: {scratch}")
    try:
        with tempfile.NamedTemporaryFile(
            dir=scratch, prefix=".write-probe-", delete=True
        ) as stream:
            stream.write(b"pilot-write-probe")
            stream.flush()
    except OSError as error:
        raise ClusterPreflightError(
            f"scratch output is not writable: {scratch}: {error}"
        ) from error

    checkpoint_path = scratch / ".cluster-pilot-checkpoint-probe.pt"
    checkpoint_path.unlink(missing_ok=True)
    probe = checkpoint_probe or _torch_checkpoint_probe
    try:
        probe(checkpoint_path)
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise ClusterPreflightError("checkpoint write/reload probe did not produce a file")
    finally:
        checkpoint_path.unlink(missing_ok=True)

    dataset = manifest["dataset"]
    return {
        "schema": "alexdoor_xas.cluster_pilot_preflight_report.v1",
        "status": "PASS",
        "python": python_version,
        "dependencies": dict(sorted(dependencies.items())),
        "isaac_modules": {},
        "source_git_commit": manifest["source_git"]["commit"],
        "manifest_schema": manifest["schema"],
        "dataset": {
            "task": dataset["task"],
            "version": dataset["version"],
            "obs_preset": dataset["obs_preset"],
            "counts": dataset["counts"],
            "split_fingerprint_sha256": dataset["split_fingerprint_sha256"],
            "spaces": dataset["spaces"],
        },
        "scratch_output": str(scratch),
        "checkpoint_round_trip": "PASS",
        "wandb_mode": config.training.wandb_mode,
        "cuda_probe": "NOT_RUN",
    }


def probe_cuda_device(
    torch_module: Any,
    *,
    expected_device_count: int,
    require_a100_80gb: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fail unless Slurm exposes exactly one usable allocated CUDA device."""
    if expected_device_count != 1:
        raise ClusterPreflightError("the pilot contract requires exactly one CUDA device")
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise ClusterPreflightError("CUDA is not available")
    count = int(cuda.device_count())
    if count != expected_device_count:
        raise ClusterPreflightError(
            f"expected exactly {expected_device_count} visible CUDA device, got {count}"
        )
    env = os.environ if environ is None else environ
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if len(visible_ids) != expected_device_count:
            raise ClusterPreflightError(
                "CUDA_VISIBLE_DEVICES does not expose exactly the allocated device"
            )
    slurm_gpus_per_task = env.get("SLURM_GPUS_PER_TASK")
    if slurm_gpus_per_task is not None and int(slurm_gpus_per_task) != expected_device_count:
        raise ClusterPreflightError("SLURM_GPUS_PER_TASK disagrees with the one-GPU pilot")
    current = int(cuda.current_device())
    if current != 0:
        raise ClusterPreflightError(f"allocated CUDA device must map to cuda:0, got cuda:{current}")
    name = str(cuda.get_device_name(0))
    properties = cuda.get_device_properties(0)
    memory_bytes = int(properties.total_memory)
    if require_a100_80gb:
        minimum_80gb_class_bytes = 75 * 1024**3
        if "A100" not in name.upper() or memory_bytes < minimum_80gb_class_bytes:
            raise ClusterPreflightError(
                f"requested A100-80GB compatibility but allocated {name!r} "
                f"with {memory_bytes / 1024**3:.1f} GiB"
            )
    tensor = torch_module.zeros(1, device="cuda:0")
    if tensor.device.type != "cuda" or (tensor.device.index or 0) != 0:
        raise ClusterPreflightError("CUDA allocation did not land on the allocated cuda:0")
    return {
        "status": "PASS",
        "visible_device_count": count,
        "current_device": current,
        "device_name": name,
        "total_memory_bytes": memory_bytes,
        "require_a100_80gb": require_a100_80gb,
        "cuda_visible_devices": visible,
        "slurm_gpus_per_task": slurm_gpus_per_task,
    }


def installed_requirements_lock() -> list[str]:
    """Build a credentials-free resolved lock as normalized name/version pairs."""
    rows = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name") and distribution.version
    }
    return sorted(rows, key=str.lower)


def write_environment_reports(
    directory: str | Path,
    *,
    dependencies: dict[str, str],
    cuda_report: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Write a sanitized resolved inventory and package lock for return to Ubuntu."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "environment_inventory.json"
    lock_path = root / "requirements.lock"
    inventory = {
        "schema": "alexdoor_xas.cluster_environment_inventory.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dict(sorted(dependencies.items())),
        "cuda": cuda_report,
    }
    _atomic_write(inventory_path, json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    _atomic_write(lock_path, "\n".join(installed_requirements_lock()) + "\n")
    return inventory_path, lock_path


def _torch_checkpoint_probe(path: Path) -> None:
    import torch

    expected = torch.arange(8, dtype=torch.float32)
    torch.save({"probe": expected}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(loaded["probe"], expected):
        raise ClusterPreflightError("checkpoint write/reload contents do not match")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ClusterPreflightError",
    "dependency_inventory",
    "installed_requirements_lock",
    "isaac_module_inventory",
    "probe_cuda_device",
    "run_pure_preflight",
    "write_environment_reports",
]
