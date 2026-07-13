"""Strict versioned configuration for the full nested dataset-scale sweep."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "alexdoor_xas.cluster_sweep_config.v1"
EXPECTED_POSES = ("D0", "D1", "D2", "D3", "D4")
EXPECTED_SPACES = ("A2_ee_delta", "A3_obj_rel_ee_delta")
EXPECTED_VIEWS = (
    ("v3_scale_n50", 50, 25, 25, 10),
    ("v3_scale_n100", 100, 25, 25, 20),
    ("v3_scale_n250", 250, 25, 25, 50),
    ("v3_scale_n500", 500, 25, 25, 100),
)
SUPPORTED_CONDA_PREFIX = "envs/alexdoor-gilbreth-pilot-py311"
SUPPORTED_NUMPY_VERSION = "2.4.6"
SUPPORTED_TORCH_VERSION = "2.12.1+cu126"
SUPPORTED_TORCH_CUDA_VERSION = "12.6"


class SweepConfigError(ValueError):
    """Raised when the checked-in scientific or execution contract drifts."""


@dataclass(frozen=True)
class SweepDatasetConfig:
    task: str
    master_version: str
    obs_preset: str
    pose_ids: tuple[str, ...]
    master_count: int
    episodes_per_pose: int
    spaces: dict[str, str]
    master_manifest: str
    views_root: str
    normalization_relative: str
    paired_source_episode_ids: bool


@dataclass(frozen=True)
class SweepSelectionConfig:
    algorithm: str
    seed: int
    holdout_per_pose_per_split: int
    pose_plan: str
    canonical_pose_plan: str
    calibration: str


@dataclass(frozen=True)
class SweepView:
    view_id: str
    train: int
    val: int
    test: int
    train_per_pose: int


@dataclass(frozen=True)
class SweepTrainingConfig:
    seed: int
    device: str
    overfit_episodes: None
    wandb_mode: str
    distributed: bool
    policies: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SweepCell:
    index: int
    policy: str
    space: str
    view_id: str
    run_id: str
    entrypoint: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class SweepStorageConfig:
    depot_root: None
    scratch_root: None
    conda_prefix_relative: str
    source_checkout_relative: str
    durable_results_relative: str
    scratch_runs_relative: str
    attempt_layout: str


@dataclass(frozen=True)
class SweepEnvironmentConfig:
    python_major_minor: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str
    specification: str
    torch_package_spec: None
    torch_index_url: None
    require_no_isaac: bool


@dataclass(frozen=True)
class SweepSlurmConfig:
    account: None
    partition: None
    qos: None
    memory: str
    cpus_per_task: int
    gpus_per_task: int
    wall_time: str
    array_max_concurrent: int
    require_a100_80gb: bool


@dataclass(frozen=True)
class SweepConfig:
    schema: str
    sweep_id: str
    dataset: SweepDatasetConfig
    selection: SweepSelectionConfig
    views: tuple[SweepView, ...]
    training: SweepTrainingConfig
    cells: tuple[SweepCell, ...]
    storage: SweepStorageConfig
    environment: SweepEnvironmentConfig
    slurm: SweepSlurmConfig
    boundaries: dict[str, bool]
    tracked_transfer_files: tuple[str, ...]
    source_path: Path


def load_sweep_config(path: str | Path) -> SweepConfig:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SweepConfigError(f"cannot load sweep config {source}: {error}") from error
    root = _mapping("root", payload)
    _keys(
        "root",
        root,
        {
            "schema",
            "sweep_id",
            "dataset",
            "selection",
            "views",
            "training",
            "cells",
            "storage",
            "environment",
            "slurm",
            "boundaries",
            "tracked_transfer_files",
        },
    )
    if root["schema"] != CONFIG_SCHEMA:
        raise SweepConfigError(f"schema must be {CONFIG_SCHEMA!r}")
    dataset = _dataset(_mapping("dataset", root["dataset"]))
    selection = _selection(_mapping("selection", root["selection"]))
    views = _views(root["views"])
    training = _training(_mapping("training", root["training"]))
    cells = _cells(root["cells"], views, training)
    storage = _storage(_mapping("storage", root["storage"]))
    environment = _environment(_mapping("environment", root["environment"]))
    slurm = _slurm(_mapping("slurm", root["slurm"]))
    boundaries = _boundaries(_mapping("boundaries", root["boundaries"]))
    tracked = _tracked(root["tracked_transfer_files"])
    return SweepConfig(
        schema=CONFIG_SCHEMA,
        sweep_id=_text("sweep_id", root["sweep_id"]),
        dataset=dataset,
        selection=selection,
        views=views,
        training=training,
        cells=cells,
        storage=storage,
        environment=environment,
        slurm=slurm,
        boundaries=boundaries,
        tracked_transfer_files=tracked,
        source_path=source,
    )


def _dataset(node: dict[str, Any]) -> SweepDatasetConfig:
    expected = {
        "task",
        "master_version",
        "obs_preset",
        "pose_ids",
        "master_count",
        "episodes_per_pose",
        "spaces",
        "master_manifest",
        "views_root",
        "normalization_relative",
        "paired_source_episode_ids",
    }
    _keys("dataset", node, expected)
    if node["task"] != "door_push_alex_v2":
        raise SweepConfigError("dataset.task must be 'door_push_alex_v2'")
    if node["master_version"] != "v3_scale_master":
        raise SweepConfigError("dataset.master_version must be 'v3_scale_master'")
    if node["obs_preset"] != "core_door_pose":
        raise SweepConfigError("dataset.obs_preset must be 'core_door_pose'")
    if tuple(node["pose_ids"]) != EXPECTED_POSES:
        raise SweepConfigError(f"dataset.pose_ids must be exactly {EXPECTED_POSES}")
    if node["master_count"] != 550:
        raise SweepConfigError("dataset.master_count must be exactly 550")
    if node["episodes_per_pose"] != 110:
        raise SweepConfigError("dataset.episodes_per_pose must be exactly 110")
    spaces_node = _mapping("dataset.spaces", node["spaces"])
    if tuple(spaces_node) != EXPECTED_SPACES:
        raise SweepConfigError(f"dataset.spaces must be ordered exactly as {EXPECTED_SPACES}")
    spaces = {
        name: _relative(f"dataset.spaces.{name}", spaces_node[name])
        for name in EXPECTED_SPACES
    }
    if node["paired_source_episode_ids"] is not True:
        raise SweepConfigError("A2/A3 source episode IDs must be paired")
    normalization = _text("dataset.normalization_relative", node["normalization_relative"])
    if normalization != "views/{view_id}/norm_stats.json":
        raise SweepConfigError("dataset.normalization_relative contract drifted")
    return SweepDatasetConfig(
        task="door_push_alex_v2",
        master_version="v3_scale_master",
        obs_preset="core_door_pose",
        pose_ids=EXPECTED_POSES,
        master_count=550,
        episodes_per_pose=110,
        spaces=spaces,
        master_manifest=_relative("dataset.master_manifest", node["master_manifest"]),
        views_root=_relative("dataset.views_root", node["views_root"]),
        normalization_relative=normalization,
        paired_source_episode_ids=True,
    )


def _selection(node: dict[str, Any]) -> SweepSelectionConfig:
    _keys(
        "selection",
        node,
        {
            "algorithm",
            "seed",
            "holdout_per_pose_per_split",
            "pose_plan",
            "canonical_pose_plan",
            "calibration",
        },
    )
    from alexdoor_xas.dataset.views import SELECTION_ALGORITHM

    if node["algorithm"] != SELECTION_ALGORITHM:
        raise SweepConfigError("selection.algorithm drifted")
    seed = _nonnegative("selection.seed", node["seed"])
    if node["holdout_per_pose_per_split"] != 5:
        raise SweepConfigError("selection holdouts must be exactly five per pose and split")
    expected_paths = {
        "pose_plan": "configs/door_pose_plan_v3_scale.json",
        "canonical_pose_plan": "configs/door_pose_plan_v2_pose.json",
        "calibration": "configs/alex_v2_door_calibration.v0.json",
    }
    selected_paths = {
        name: _relative(f"selection.{name}", node[name]) for name in expected_paths
    }
    if selected_paths != expected_paths:
        raise SweepConfigError("selection pose plan, canonical geometry, or calibration drifted")
    return SweepSelectionConfig(
        SELECTION_ALGORITHM,
        seed,
        5,
        selected_paths["pose_plan"],
        selected_paths["canonical_pose_plan"],
        selected_paths["calibration"],
    )


def _views(value: Any) -> tuple[SweepView, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise SweepConfigError("views must contain exactly four view contracts")
    result: list[SweepView] = []
    for index, raw in enumerate(value):
        node = _mapping(f"views[{index}]", raw)
        _keys(f"views[{index}]", node, {"view_id", "train", "val", "test", "train_per_pose"})
        result.append(
            SweepView(
                _text("view_id", node["view_id"]),
                _positive("view.train", node["train"]),
                _positive("view.val", node["val"]),
                _positive("view.test", node["test"]),
                _positive("view.train_per_pose", node["train_per_pose"]),
            )
        )
    actual = tuple((v.view_id, v.train, v.val, v.test, v.train_per_pose) for v in result)
    if actual != EXPECTED_VIEWS:
        raise SweepConfigError(f"view contracts must be exactly {EXPECTED_VIEWS}")
    return tuple(result)


def _training(node: dict[str, Any]) -> SweepTrainingConfig:
    _keys(
        "training",
        node,
        {"seed", "device", "overfit_episodes", "wandb_mode", "distributed", "policies"},
    )
    if node["seed"] != 0 or node["device"] != "cuda":
        raise SweepConfigError("training must use seed 0 and CUDA")
    if node["overfit_episodes"] is not None:
        raise SweepConfigError(
            "training.overfit_episodes must remain null; pilot overfit is forbidden"
        )
    if node["wandb_mode"] != "offline":
        raise SweepConfigError("training.wandb_mode must be explicitly offline")
    if node["distributed"] is not False:
        raise SweepConfigError("distributed training is forbidden; each cell uses one GPU")
    policies = _mapping("training.policies", node["policies"])
    expected_policies = {
        "act": {"epochs": 100, "val_every": 5},
        "diffusion": {
            "epochs": 300,
            "val_every": 10,
            "use_ema": True,
            "val_inference_steps": 10,
        },
    }
    if policies != expected_policies:
        raise SweepConfigError("training policy defaults drifted from committed non-pilot defaults")
    return SweepTrainingConfig(0, "cuda", None, "offline", False, policies)


def _cells(
    value: Any, views: tuple[SweepView, ...], training: SweepTrainingConfig
) -> tuple[SweepCell, ...]:
    if not isinstance(value, list) or len(value) != 16:
        raise SweepConfigError("cells must contain exactly 16 entries")
    cells: list[SweepCell] = []
    for index, raw in enumerate(value):
        node = _mapping(f"cells[{index}]", raw)
        _keys(
            f"cells[{index}]",
            node,
            {"index", "policy", "space", "view_id", "run_id", "entrypoint", "overrides"},
        )
        cells.append(
            SweepCell(
                _nonnegative("cell.index", node["index"]),
                _text("cell.policy", node["policy"]),
                _text("cell.space", node["space"]),
                _text("cell.view_id", node["view_id"]),
                _text("cell.run_id", node["run_id"]),
                _relative("cell.entrypoint", node["entrypoint"]),
                dict(_mapping("cell.overrides", node["overrides"])),
            )
        )
    expected: list[tuple[Any, ...]] = []
    cell_index = 0
    for view in views:
        n = view.train
        for policy, space in (
            ("act", "A2_ee_delta"),
            ("act", "A3_obj_rel_ee_delta"),
            ("diffusion", "A2_ee_delta"),
            ("diffusion", "A3_obj_rel_ee_delta"),
        ):
            short_space = "a2" if space.startswith("A2") else "a3"
            run_id = f"sweep_{policy}_{short_space}_n{n}_seed0"
            entrypoint = f"scripts/train_{policy}.py"
            overrides = {f"train.{key}": val for key, val in training.policies[policy].items()}
            expected.append(
                (
                    cell_index,
                    policy,
                    space,
                    view.view_id,
                    run_id,
                    entrypoint,
                    overrides,
                )
            )
            cell_index += 1
    actual = [
        (c.index, c.policy, c.space, c.view_id, c.run_id, c.entrypoint, c.overrides)
        for c in cells
    ]
    if actual != expected:
        raise SweepConfigError(
            "cell identity, run_id, override, or stable task-index mapping drifted"
        )
    return tuple(cells)


def _storage(node: dict[str, Any]) -> SweepStorageConfig:
    fields = {
        "depot_root",
        "scratch_root",
        "conda_prefix_relative",
        "source_checkout_relative",
        "durable_results_relative",
        "scratch_runs_relative",
        "attempt_layout",
    }
    _keys("storage", node, fields)
    if node["depot_root"] is not None or node["scratch_root"] is not None:
        raise SweepConfigError("checked-in depot_root and scratch_root must remain null")
    layout = _text("storage.attempt_layout", node["attempt_layout"])
    if layout != "attempts/{slurm_array_job_id}/{slurm_array_task_id}/{run_id}":
        raise SweepConfigError("storage attempt layout drifted")
    conda_prefix = _relative(
        "storage.conda_prefix_relative", node["conda_prefix_relative"]
    )
    if conda_prefix != SUPPORTED_CONDA_PREFIX:
        raise SweepConfigError(
            "storage.conda_prefix_relative must reuse the proven pilot environment"
        )
    return SweepStorageConfig(
        None,
        None,
        conda_prefix,
        _relative("storage.source_checkout_relative", node["source_checkout_relative"]),
        _relative("storage.durable_results_relative", node["durable_results_relative"]),
        _relative("storage.scratch_runs_relative", node["scratch_runs_relative"]),
        layout,
    )


def _environment(node: dict[str, Any]) -> SweepEnvironmentConfig:
    _keys(
        "environment",
        node,
        {
            "python_major_minor",
            "numpy_version",
            "torch_version",
            "torch_cuda_version",
            "specification",
            "torch_package_spec",
            "torch_index_url",
            "require_no_isaac",
        },
    )
    if node["python_major_minor"] != "3.11":
        raise SweepConfigError("environment Python must be 3.11")
    if node["numpy_version"] != SUPPORTED_NUMPY_VERSION:
        raise SweepConfigError(f"environment NumPy must be {SUPPORTED_NUMPY_VERSION}")
    if node["torch_version"] != SUPPORTED_TORCH_VERSION:
        raise SweepConfigError(f"environment PyTorch must be {SUPPORTED_TORCH_VERSION}")
    if node["torch_cuda_version"] != SUPPORTED_TORCH_CUDA_VERSION:
        raise SweepConfigError(
            f"environment Torch CUDA must be {SUPPORTED_TORCH_CUDA_VERSION}"
        )
    if node["torch_package_spec"] is not None or node["torch_index_url"] is not None:
        raise SweepConfigError(
            "checked-in PyTorch build must remain unset pending live cluster evidence"
        )
    if node["require_no_isaac"] is not True:
        raise SweepConfigError("cluster environment must require no Isaac")
    return SweepEnvironmentConfig(
        "3.11",
        SUPPORTED_NUMPY_VERSION,
        SUPPORTED_TORCH_VERSION,
        SUPPORTED_TORCH_CUDA_VERSION,
        _relative("environment.specification", node["specification"]),
        None,
        None,
        True,
    )


def _slurm(node: dict[str, Any]) -> SweepSlurmConfig:
    _keys(
        "slurm",
        node,
        {
            "account",
            "partition",
            "qos",
            "memory",
            "cpus_per_task",
            "gpus_per_task",
            "wall_time",
            "array_max_concurrent",
            "require_a100_80gb",
        },
    )
    if any(node[name] is not None for name in ("account", "partition", "qos")):
        raise SweepConfigError("checked-in scheduler account/partition/qos must remain null")
    if node["gpus_per_task"] != 1:
        raise SweepConfigError("each Slurm cell must request exactly one GPU")
    if node["array_max_concurrent"] != 2:
        raise SweepConfigError("checked-in Slurm concurrency must default to 2")
    if not isinstance(node["require_a100_80gb"], bool):
        raise SweepConfigError("slurm.require_a100_80gb must be boolean")
    return SweepSlurmConfig(
        None,
        None,
        None,
        _text("slurm.memory", node["memory"]),
        _positive("slurm.cpus_per_task", node["cpus_per_task"]),
        1,
        _text("slurm.wall_time", node["wall_time"]),
        2,
        node["require_a100_80gb"],
    )


def _boundaries(node: dict[str, Any]) -> dict[str, bool]:
    expected = {
        "cluster_dataset_generation": False,
        "cluster_isaac": False,
        "cluster_closed_loop_evaluation": False,
        "transfer_authorized": False,
        "submission_authorized": False,
        "phase4_authorized": False,
    }
    if node != expected:
        raise SweepConfigError("no-Isaac/no-transfer/no-submit/no-Phase-4 boundaries drifted")
    return expected


def _tracked(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SweepConfigError("tracked_transfer_files must be a non-empty list")
    paths = tuple(_relative("tracked_transfer_files", item) for item in value)
    if len(paths) != len(set(paths)):
        raise SweepConfigError("tracked_transfer_files contains duplicates")
    required = {
        "configs/cluster_sweep.v1.json",
        "configs/door_pose_plan_v3_scale.json",
        "configs/door_pose_plan_v2_pose.json",
        "configs/alex_v2_door_calibration.v0.json",
        "environment/gilbreth_pilot_py311.yml",
        "scripts/build_cluster_sweep_manifest.py",
        "scripts/preflight_cluster_sweep.py",
        "scripts/render_cluster_sweep_slurm.py",
        "scripts/build_cluster_sweep_return_manifest.py",
        "scripts/verify_returned_cluster_sweep.py",
        "src/alexdoor_xas/cluster_sweep/config.py",
        "src/alexdoor_xas/cluster_sweep/preflight.py",
        "src/alexdoor_xas/cluster_sweep/returns.py",
        "src/alexdoor_xas/cluster_sweep/slurm.py",
        "src/alexdoor_xas/cluster_sweep/transfer.py",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise SweepConfigError(f"tracked_transfer_files missing required paths: {missing}")
    return paths


def canonical_resolved_config_sha256(resolved_config: Mapping[str, Any]) -> str:
    """Hash one resolved trainer configuration with the training-time algorithm."""
    if not isinstance(resolved_config, Mapping):
        raise SweepConfigError("resolved training config must be a mapping")
    try:
        canonical = json.dumps(
            dict(resolved_config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise SweepConfigError(
            f"resolved training config is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(canonical).hexdigest()


def sweep_cell_override_values(
    config: SweepConfig,
    cell: SweepCell,
    *,
    output_root: str,
    wandb_dir: str,
) -> dict[str, Any]:
    """Return the one authoritative Hydra override mapping for a sweep cell."""
    if cell not in config.cells:
        raise SweepConfigError("sweep cell is not part of the loaded configuration")
    if not output_root or not wandb_dir:
        raise SweepConfigError("sweep runtime output and W&B directories must be non-empty")
    return {
        "dataset.task": config.dataset.task,
        "dataset.space": cell.space,
        "dataset.version": config.dataset.master_version,
        "dataset.view_id": cell.view_id,
        "dataset.obs_preset": config.dataset.obs_preset,
        "train.seed": config.training.seed,
        "train.device": config.training.device,
        "train.overfit_episodes": None,
        "run.run_id": cell.run_id,
        "run.output_root": output_root,
        "+wandb.mode": config.training.wandb_mode,
        "+wandb.dir": wandb_dir,
        **cell.overrides,
    }


def resolved_sweep_cell_config(
    config: SweepConfig,
    cell: SweepCell,
    *,
    output_root: str,
    wandb_dir: str,
) -> dict[str, Any]:
    """Compose one cell through the same ACT/Diffusion config loaders used for training."""
    values = sweep_cell_override_values(
        config, cell, output_root=output_root, wandb_dir=wandb_dir
    )
    overrides = [f"{key}={_override_text(value)}" for key, value in values.items()]
    if cell.policy == "act":
        from alexdoor_xas.policies.act.config import load_act_config

        resolved = load_act_config(overrides)
    elif cell.policy == "diffusion":
        from alexdoor_xas.policies.diffusion.config import load_diffusion_config

        resolved = load_diffusion_config(overrides)
    else:  # pragma: no cover - load_sweep_config already freezes this inventory.
        raise SweepConfigError(f"unsupported sweep policy: {cell.policy}")
    return asdict(resolved)


def validate_resolved_sweep_cell_config(
    config: SweepConfig,
    cell: SweepCell,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact equality with the normal resolved config for this cell."""
    resolved = dict(resolved_config)
    run = resolved.get("run")
    wandb = resolved.get("wandb_overrides")
    if not isinstance(run, dict) or not isinstance(wandb, dict):
        raise SweepConfigError("resolved sweep config lacks run or W&B mappings")
    output_root = run.get("output_root")
    wandb_dir = wandb.get("dir")
    if not isinstance(output_root, str) or not output_root:
        raise SweepConfigError("resolved sweep config has no runtime output root")
    if not isinstance(wandb_dir, str) or not wandb_dir:
        raise SweepConfigError("resolved sweep config has no W&B directory")
    expected = resolved_sweep_cell_config(
        config,
        cell,
        output_root=output_root,
        wandb_dir=wandb_dir,
    )
    if resolved != expected:
        raise SweepConfigError(
            f"resolved config does not equal configured sweep cell {cell.run_id}: "
            f"{_first_difference(expected, resolved)}"
        )
    return expected


def _override_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _first_difference(expected: Any, actual: Any, path: str = "root") -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return f"{path} keys expected={sorted(expected)} actual={sorted(actual)}"
        for key in expected:
            if expected[key] != actual[key]:
                return _first_difference(expected[key], actual[key], f"{path}.{key}")
    return f"{path} expected={expected!r} actual={actual!r}"


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SweepConfigError(f"{name} must be an object")
    return value


def _keys(name: str, node: Mapping[str, Any], expected: set[str]) -> None:
    missing = sorted(expected - set(node))
    extra = sorted(set(node) - expected)
    if missing or extra:
        raise SweepConfigError(f"{name} keys mismatch: missing={missing}, extra={extra}")


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in ("\n", "\r", "\0")):
        raise SweepConfigError(f"{name} must be a safe non-empty string")
    return value


def _relative(name: str, value: Any) -> str:
    text = _text(name, value)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise SweepConfigError(f"{name} must be a normalized repository-relative path")
    return text


def _positive(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SweepConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SweepConfigError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "CONFIG_SCHEMA",
    "EXPECTED_POSES",
    "EXPECTED_SPACES",
    "EXPECTED_VIEWS",
    "SUPPORTED_CONDA_PREFIX",
    "SUPPORTED_NUMPY_VERSION",
    "SUPPORTED_TORCH_CUDA_VERSION",
    "SUPPORTED_TORCH_VERSION",
    "SweepCell",
    "SweepConfig",
    "SweepConfigError",
    "SweepDatasetConfig",
    "SweepEnvironmentConfig",
    "SweepSelectionConfig",
    "SweepSlurmConfig",
    "SweepStorageConfig",
    "SweepTrainingConfig",
    "SweepView",
    "canonical_resolved_config_sha256",
    "load_sweep_config",
    "resolved_sweep_cell_config",
    "sweep_cell_override_values",
    "validate_resolved_sweep_cell_config",
]
