"""Validated configuration contract for the Gilbreth N50 compatibility pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "alexdoor_xas.cluster_pilot_config.v1"
EXPECTED_TASK = "door_push_alex_v2"
EXPECTED_VERSION = "v2_pose"
EXPECTED_SPACES = ("A2_ee_delta", "A3_obj_rel_ee_delta")
EXPECTED_COUNTS = {"total": 50, "train": 38, "val": 6, "test": 6}
EXPECTED_OBS_PRESET = "core_door_pose"


class PilotConfigError(ValueError):
    """Raised when the checked-in pilot contract drifts from the approved design."""


@dataclass(frozen=True)
class SourceDatasetConfig:
    task: str
    version: str
    spaces: dict[str, str]
    shared_split: str
    counts: dict[str, int]
    obs_preset: str


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    device: str
    epochs: int
    val_every: int
    overfit_episodes: int
    wandb_mode: str


@dataclass(frozen=True)
class PilotCell:
    index: int
    policy: str
    space: str
    run_id: str
    entrypoint: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class StorageConfig:
    depot_root: str | None
    scratch_root: str | None
    conda_prefix_relative: str
    source_checkout_relative: str
    durable_results_relative: str
    scratch_runs_relative: str


@dataclass(frozen=True)
class EnvironmentConfig:
    python_major_minor: str
    specification: str
    torch_package_spec: str | None
    torch_index_url: str | None
    require_no_isaac: bool


@dataclass(frozen=True)
class SlurmConfig:
    account: str | None
    partition: str | None
    qos: str | None
    memory: str
    cpus_per_task: int
    gpus_per_node: int
    wall_time: str
    array_max_concurrent: int
    require_a100_80gb: bool


@dataclass(frozen=True)
class PilotConfig:
    schema: str
    pilot_id: str
    source_dataset: SourceDatasetConfig
    training: TrainingConfig
    cells: tuple[PilotCell, ...]
    storage: StorageConfig
    environment: EnvironmentConfig
    slurm: SlurmConfig
    tracked_transfer_files: tuple[str, ...]
    future_sweep: dict[str, Any]
    source_path: Path


def load_pilot_config(path: str | Path) -> PilotConfig:
    """Load and fail-closed validate the versioned pilot JSON."""
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PilotConfigError(f"cannot load pilot config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise PilotConfigError("pilot config must be a JSON object")

    _require_keys(
        "root",
        payload,
        {
            "schema",
            "pilot_id",
            "source_dataset",
            "training",
            "cells",
            "storage",
            "environment",
            "slurm",
            "tracked_transfer_files",
            "future_sweep",
        },
    )
    if payload["schema"] != CONFIG_SCHEMA:
        raise PilotConfigError(f"schema must be {CONFIG_SCHEMA!r}")
    pilot_id = _required_string("pilot_id", payload["pilot_id"])

    dataset = _build_dataset(_mapping("source_dataset", payload["source_dataset"]))
    training = _build_training(_mapping("training", payload["training"]))
    cells = _build_cells(payload["cells"])
    storage = _build_storage(_mapping("storage", payload["storage"]))
    environment = _build_environment(_mapping("environment", payload["environment"]))
    slurm = _build_slurm(_mapping("slurm", payload["slurm"]))
    tracked = _build_tracked_files(payload["tracked_transfer_files"])
    future = _build_future_sweep(_mapping("future_sweep", payload["future_sweep"]))

    return PilotConfig(
        schema=payload["schema"],
        pilot_id=pilot_id,
        source_dataset=dataset,
        training=training,
        cells=cells,
        storage=storage,
        environment=environment,
        slurm=slurm,
        tracked_transfer_files=tracked,
        future_sweep=future,
        source_path=source,
    )


def _build_dataset(node: dict[str, Any]) -> SourceDatasetConfig:
    _require_keys(
        "source_dataset",
        node,
        {"task", "version", "spaces", "shared_split", "expected_counts", "obs_preset"},
    )
    if node["task"] != EXPECTED_TASK:
        raise PilotConfigError(f"source_dataset.task must be {EXPECTED_TASK!r}")
    if node["version"] != EXPECTED_VERSION:
        raise PilotConfigError(f"source_dataset.version must be {EXPECTED_VERSION!r}")
    spaces_node = _mapping("source_dataset.spaces", node["spaces"])
    if tuple(spaces_node) != EXPECTED_SPACES:
        raise PilotConfigError(
            f"source_dataset.spaces must be ordered exactly as {list(EXPECTED_SPACES)!r}"
        )
    spaces = {
        space: _relative_path(f"source_dataset.spaces.{space}", spaces_node[space])
        for space in EXPECTED_SPACES
    }
    counts_node = _mapping("source_dataset.expected_counts", node["expected_counts"])
    counts = {
        name: _positive_int(f"expected_counts.{name}", counts_node.get(name))
        for name in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS or set(counts_node) != set(EXPECTED_COUNTS):
        raise PilotConfigError(f"expected_counts must be exactly {EXPECTED_COUNTS}")
    if node["obs_preset"] != EXPECTED_OBS_PRESET:
        raise PilotConfigError(f"obs_preset must be {EXPECTED_OBS_PRESET!r}")
    shared_split = _relative_path("source_dataset.shared_split", node["shared_split"])
    return SourceDatasetConfig(
        task=EXPECTED_TASK,
        version=EXPECTED_VERSION,
        spaces=spaces,
        shared_split=shared_split,
        counts=counts,
        obs_preset=EXPECTED_OBS_PRESET,
    )


def _build_training(node: dict[str, Any]) -> TrainingConfig:
    _require_keys(
        "training",
        node,
        {"seed", "device", "epochs", "val_every", "overfit_episodes", "wandb_mode"},
    )
    expected = {
        "seed": 0,
        "device": "cuda",
        "epochs": 2,
        "val_every": 1,
        "overfit_episodes": 2,
        "wandb_mode": "offline",
    }
    for name, value in expected.items():
        if node[name] != value:
            raise PilotConfigError(f"training.{name} must be {value!r}")
    return TrainingConfig(**expected)


def _build_cells(value: Any) -> tuple[PilotCell, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise PilotConfigError("cells must contain exactly two pilot cells")
    cells: list[PilotCell] = []
    for index, raw in enumerate(value):
        node = _mapping(f"cells[{index}]", raw)
        _require_keys(
            f"cells[{index}]",
            node,
            {"index", "policy", "space", "run_id", "entrypoint", "overrides"},
        )
        overrides = _mapping(f"cells[{index}].overrides", node["overrides"])
        cells.append(
            PilotCell(
                index=_nonnegative_int(f"cells[{index}].index", node["index"]),
                policy=_required_string(f"cells[{index}].policy", node["policy"]),
                space=_required_string(f"cells[{index}].space", node["space"]),
                run_id=_required_string(f"cells[{index}].run_id", node["run_id"]),
                entrypoint=_relative_path(f"cells[{index}].entrypoint", node["entrypoint"]),
                overrides=dict(overrides),
            )
        )
    expected_identity = [
        (0, "act", "A2_ee_delta", "pilot_act_a2_n50_seed0", "scripts/train_act.py"),
        (
            1,
            "diffusion",
            "A3_obj_rel_ee_delta",
            "pilot_diffusion_a3_n50_seed0",
            "scripts/train_diffusion.py",
        ),
    ]
    actual_identity = [
        (cell.index, cell.policy, cell.space, cell.run_id, cell.entrypoint) for cell in cells
    ]
    if actual_identity != expected_identity:
        raise PilotConfigError("cells must be ACT-on-A2 then Diffusion-on-A3 with approved IDs")
    if cells[0].overrides != {"model.chunk_size": 40}:
        raise PilotConfigError("ACT pilot must set model.chunk_size=40")
    expected_diffusion = {
        "model.horizon": 16,
        "train.use_ema": True,
        "train.val_inference_steps": 5,
        "rollout.sampler": "ddim",
        "rollout.num_inference_steps": 5,
    }
    if cells[1].overrides != expected_diffusion:
        raise PilotConfigError("Diffusion pilot overrides drifted from the approved short cell")
    return tuple(cells)


def _build_storage(node: dict[str, Any]) -> StorageConfig:
    _require_keys(
        "storage",
        node,
        {
            "depot_root",
            "scratch_root",
            "conda_prefix_relative",
            "source_checkout_relative",
            "durable_results_relative",
            "scratch_runs_relative",
        },
    )
    if node["depot_root"] is not None or node["scratch_root"] is not None:
        raise PilotConfigError("checked-in depot_root and scratch_root must remain null")
    return StorageConfig(
        depot_root=None,
        scratch_root=None,
        conda_prefix_relative=_relative_path(
            "storage.conda_prefix_relative", node["conda_prefix_relative"]
        ),
        source_checkout_relative=_relative_path(
            "storage.source_checkout_relative", node["source_checkout_relative"]
        ),
        durable_results_relative=_relative_path(
            "storage.durable_results_relative", node["durable_results_relative"]
        ),
        scratch_runs_relative=_relative_path(
            "storage.scratch_runs_relative", node["scratch_runs_relative"]
        ),
    )


def _build_environment(node: dict[str, Any]) -> EnvironmentConfig:
    _require_keys(
        "environment",
        node,
        {
            "python_major_minor",
            "specification",
            "torch_package_spec",
            "torch_index_url",
            "require_no_isaac",
        },
    )
    if node["python_major_minor"] != "3.11":
        raise PilotConfigError("environment.python_major_minor must be '3.11'")
    if node["torch_package_spec"] is not None or node["torch_index_url"] is not None:
        raise PilotConfigError(
            "checked-in PyTorch package/build and index must remain unset until Gilbreth evidence"
        )
    if node["require_no_isaac"] is not True:
        raise PilotConfigError("environment.require_no_isaac must be true")
    return EnvironmentConfig(
        python_major_minor="3.11",
        specification=_relative_path("environment.specification", node["specification"]),
        torch_package_spec=None,
        torch_index_url=None,
        require_no_isaac=True,
    )


def _build_slurm(node: dict[str, Any]) -> SlurmConfig:
    _require_keys(
        "slurm",
        node,
        {
            "account",
            "partition",
            "qos",
            "memory",
            "cpus_per_task",
            "gpus_per_node",
            "wall_time",
            "array_max_concurrent",
            "require_a100_80gb",
        },
    )
    for name in ("account", "partition", "qos"):
        if node[name] is not None:
            raise PilotConfigError(f"checked-in slurm.{name} must remain null")
    memory = _required_string("slurm.memory", node["memory"])
    wall_time = _required_string("slurm.wall_time", node["wall_time"])
    cpus = _positive_int("slurm.cpus_per_task", node["cpus_per_task"])
    gpus = _positive_int("slurm.gpus_per_node", node["gpus_per_node"])
    concurrent = _positive_int("slurm.array_max_concurrent", node["array_max_concurrent"])
    if gpus != 1:
        raise PilotConfigError("slurm.gpus_per_node must be exactly 1")
    if concurrent > 2:
        raise PilotConfigError("slurm.array_max_concurrent must be at most 2")
    if not isinstance(node["require_a100_80gb"], bool):
        raise PilotConfigError("slurm.require_a100_80gb must be boolean")
    return SlurmConfig(
        account=None,
        partition=None,
        qos=None,
        memory=memory,
        cpus_per_task=cpus,
        gpus_per_node=gpus,
        wall_time=wall_time,
        array_max_concurrent=concurrent,
        require_a100_80gb=node["require_a100_80gb"],
    )


def _build_tracked_files(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PilotConfigError("tracked_transfer_files must be a non-empty list")
    paths = tuple(_relative_path("tracked_transfer_files", item) for item in value)
    if len(paths) != len(set(paths)):
        raise PilotConfigError("tracked_transfer_files contains duplicates")
    required = {
        "knowledge/wiki/implementation_phases/extra-03-gilbreth-compatibility-pilot.md",
        "pyproject.toml",
        "configs/act.yaml",
        "configs/diffusion.yaml",
        "configs/wandb.yaml",
        "configs/cluster_pilot_n50.v1.json",
        "environment/gilbreth_pilot_py311.yml",
        "scripts/bootstrap_gilbreth_pilot.sh",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise PilotConfigError(f"tracked_transfer_files is missing required paths: {missing}")
    return paths


def _build_future_sweep(node: dict[str, Any]) -> dict[str, Any]:
    required = {
        "status": "contract_only_do_not_generate_or_launch",
        "master_pool": "one deterministic master episode pool",
        "training_episode_counts": [50, 100, 250, 500],
        "n_definition": "number of training episodes",
        "nested_training_subsets": True,
        "fixed_shared_validation_test": True,
        "equal_pose_balance": True,
        "paired_a2_a3_source_episodes": True,
        "existing_v2_pose_role": "stabilization and compatibility-pilot evidence only",
        "full_matrix_cells": 16,
    }
    if node != required:
        raise PilotConfigError("future_sweep contract drifted from the approved nested design")
    return dict(node)


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PilotConfigError(f"{name} must be an object")
    return value


def _require_keys(name: str, node: dict[str, Any], expected: set[str]) -> None:
    missing = sorted(expected - set(node))
    extra = sorted(set(node) - expected)
    if missing or extra:
        raise PilotConfigError(f"{name} keys mismatch: missing={missing}, extra={extra}")


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotConfigError(f"{name} must be a non-empty string")
    if any(character in value for character in ("\n", "\r", "\0")):
        raise PilotConfigError(f"{name} contains a forbidden control character")
    return value


def _relative_path(name: str, value: Any) -> str:
    text = _required_string(name, value)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise PilotConfigError(f"{name} must be a normalized repository-relative path")
    return text


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PilotConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PilotConfigError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "CONFIG_SCHEMA",
    "EXPECTED_COUNTS",
    "EXPECTED_OBS_PRESET",
    "EXPECTED_SPACES",
    "EnvironmentConfig",
    "PilotCell",
    "PilotConfig",
    "PilotConfigError",
    "SlurmConfig",
    "SourceDatasetConfig",
    "StorageConfig",
    "TrainingConfig",
    "load_pilot_config",
]
