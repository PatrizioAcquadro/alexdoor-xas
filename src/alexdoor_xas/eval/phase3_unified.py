"""Fail-closed orchestration and reporting for the Phase 3 unified evaluation.

The learned-policy evaluators remain the rollout authorities.  This module
stages independent checkpoint copies, invokes one evaluator process per pose,
validates its evidence, and exclusively publishes a canonical pose artifact.
Returned Gilbreth artifacts are read-only inputs and are never execution
destinations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

PLAN_SCHEMA = "alexdoor_xas.phase3_unified_eval_plan.v1"
CANONICAL_SCHEMA = "alexdoor_xas.phase3_unified_pose_eval.v1"
COMPLETION_SCHEMA = "alexdoor_xas.phase3_unified_cell_completion.v1"
EXPECTED_SOURCE_COMMIT = "efa39434a123dab4d029f5f4ffdb122844892a6d"
EXPECTED_ATTEMPT = "11281591"
EXPECTED_POLICY_SPACE_SIZE = {
    (policy, space, size)
    for size in (50, 100, 250, 500)
    for policy in ("act", "diffusion")
    for space in ("A2_ee_delta", "A3_obj_rel_ee_delta")
}
REQUIRED_PROVENANCE_FIELDS = {
    "checkpoint_dataset_fingerprint_sha256",
    "live_dataset_fingerprint_sha256",
    "master_dataset_fingerprint_sha256",
    "action_dataset_fingerprint_sha256",
    "checkpoint_split_fingerprint_sha256",
    "split_fingerprint_sha256",
    "view_id",
    "view_fingerprint_sha256",
    "normalization_sha256",
    "split_episode_ids",
}
REQUIRED_ROW_FIELDS = {
    "seed",
    "randomized",
    "door_pose_id",
    "door_yaw_deg",
    "door_offset_xy",
    "success",
    "initial_angle_rad",
    "final_angle_rad",
    "door_angle_change_rad",
    "n_ticks",
    "first_success_tick",
    "time_to_success_s",
    "termination_reason",
    "failure_label",
    "env_truncated",
    "n_accepted",
    "n_corrected",
    "n_rejected",
    "n_warnings",
    "warning_counts",
    "warning_family_counts",
    "warning_records",
    "contact_ticks",
    "contact_source",
    "contact_unavailable_reason",
    "force_n",
    "force_n_all_samples",
    "force_trace_evidence",
    "force_exceeds_admission_bound",
    "impulse_ns",
    "policy_metadata_keys",
    "notes",
}


class UnifiedEvalError(RuntimeError):
    """A Phase 3 plan, artifact, execution, or evidence gate failed."""


@dataclass(frozen=True)
class PosePlan:
    pose_id: str
    door_yaw_deg: float
    door_offset_x_m: float
    door_offset_y_m: float
    base_seed: int
    episodes_fixed: int
    episodes_randomized: int

    @property
    def n_rollouts(self) -> int:
        return self.episodes_fixed + self.episodes_randomized

    @property
    def fixed_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.base_seed, self.base_seed + self.episodes_fixed))

    @property
    def randomized_seeds(self) -> tuple[int, ...]:
        start = self.base_seed + self.episodes_fixed
        return tuple(range(start, start + self.episodes_randomized))


@dataclass(frozen=True)
class CellPlan:
    index: int
    run_id: str
    policy: str
    action_space: str
    dataset_size: int
    view_id: str
    returned_checkpoint: Path
    workspace_checkpoint: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class UnifiedPlan:
    path: Path
    repo_root: Path
    raw: dict[str, Any]
    protocol_id: str
    source_git_commit: str
    attempt_id: str
    transfer_manifest: Path
    return_root: Path
    workspace_root: Path
    curated_root: Path
    inventory_hashes: Path
    inventory_sizes: Path
    poses: tuple[PosePlan, ...]
    cells: tuple[CellPlan, ...]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    def cell(self, run_id: str) -> CellPlan:
        matches = [cell for cell in self.cells if cell.run_id == run_id]
        if len(matches) != 1:
            raise UnifiedEvalError(f"unknown or duplicate run_id {run_id!r}")
        return matches[0]

    def pose(self, pose_id: str) -> PosePlan:
        matches = [pose for pose in self.poses if pose.pose_id == pose_id]
        if len(matches) != 1:
            raise UnifiedEvalError(f"unknown or duplicate pose_id {pose_id!r}")
        return matches[0]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise UnifiedEvalError(f"{label} must be a lowercase SHA-256")
    return text


def load_plan(
    path: str | Path = paths.REPO_ROOT / "configs" / "phase3_unified_eval.v1.json",
    *,
    repo_root: str | Path = paths.REPO_ROOT,
) -> UnifiedPlan:
    repo = Path(repo_root).resolve()
    plan_path = _resolve(repo, str(path))
    try:
        raw = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UnifiedEvalError(f"cannot load evaluation plan {plan_path}: {error}") from error
    if raw.get("schema") != PLAN_SCHEMA:
        raise UnifiedEvalError("evaluation plan schema mismatch")
    if raw.get("protocol_id") != "phase3_unified_primary_v1":
        raise UnifiedEvalError("evaluation protocol ID drifted")
    if raw.get("source_git_commit") != EXPECTED_SOURCE_COMMIT:
        raise UnifiedEvalError("evaluation source commit drifted")
    if raw.get("attempt_id") != EXPECTED_ATTEMPT:
        raise UnifiedEvalError("evaluation attempt ID drifted")

    rollout = raw.get("rollout") or {}
    pose_rows = rollout.get("poses") or []
    poses = tuple(
        PosePlan(
            pose_id=str(row["pose_id"]),
            door_yaw_deg=float(row["door_yaw_deg"]),
            door_offset_x_m=float(row["door_offset_x_m"]),
            door_offset_y_m=float(row["door_offset_y_m"]),
            base_seed=int(row["base_seed"]),
            episodes_fixed=int(row["episodes_fixed"]),
            episodes_randomized=int(row["episodes_randomized"]),
        )
        for row in pose_rows
    )
    expected_poses = {
        "D0": (0.0, 0.0, 0.0, 100, 5, 15),
        "D1": (2.8648, 0.02, 0.0, 200, 1, 3),
        "D2": (-2.8648, 0.0, -0.02, 210, 1, 3),
        "D3": (5.7296, 0.02, 0.02, 220, 1, 3),
        "D4": (-5.7296, 0.02, -0.02, 230, 1, 3),
    }
    actual_poses = {
        pose.pose_id: (
            pose.door_yaw_deg,
            pose.door_offset_x_m,
            pose.door_offset_y_m,
            pose.base_seed,
            pose.episodes_fixed,
            pose.episodes_randomized,
        )
        for pose in poses
    }
    if actual_poses != expected_poses or sum(pose.n_rollouts for pose in poses) != 36:
        raise UnifiedEvalError("D0-D4 pose/seed protocol drifted")

    return_root = _resolve(repo, raw["return_root"])
    workspace_root = _resolve(repo, raw["workspace_root"])
    curated_root = _resolve(repo, raw["curated_root"])
    if workspace_root.is_relative_to(return_root) or curated_root.is_relative_to(return_root):
        raise UnifiedEvalError("evaluation outputs may not be inside the returned package")
    cells = tuple(
        CellPlan(
            index=int(row["index"]),
            run_id=str(row["run_id"]),
            policy=str(row["policy"]),
            action_space=str(row["action_space"]),
            dataset_size=int(row["dataset_size"]),
            view_id=str(row["view_id"]),
            returned_checkpoint=_resolve(repo, row["returned_checkpoint"]),
            workspace_checkpoint=_resolve(repo, row["workspace_checkpoint"]),
            checkpoint_sha256=_require_sha256(
                row["checkpoint_sha256"], f"{row['run_id']} checkpoint"
            ),
        )
        for row in raw.get("cells") or []
    )
    if len(cells) != 16 or [cell.index for cell in cells] != list(range(16)):
        raise UnifiedEvalError("plan must contain stable cell indices 0-15")
    identities = {(cell.policy, cell.action_space, cell.dataset_size) for cell in cells}
    if identities != EXPECTED_POLICY_SPACE_SIZE or len({cell.run_id for cell in cells}) != 16:
        raise UnifiedEvalError("plan does not contain the exact 16 policy/space/size cells")
    for cell in cells:
        expected_returned = (
            return_root
            / "attempts"
            / EXPECTED_ATTEMPT
            / str(cell.index)
            / cell.run_id
            / "checkpoints"
            / "best.pt"
        )
        expected_workspace = workspace_root / "runs" / cell.run_id / "checkpoints" / "best.pt"
        if cell.returned_checkpoint != expected_returned.resolve():
            raise UnifiedEvalError(f"returned checkpoint mapping drifted: {cell.run_id}")
        if cell.workspace_checkpoint != expected_workspace.resolve():
            raise UnifiedEvalError(f"workspace checkpoint mapping drifted: {cell.run_id}")
        if not cell.view_id == f"v3_scale_n{cell.dataset_size}":
            raise UnifiedEvalError(f"dataset view/size mismatch: {cell.run_id}")

    runtime = raw.get("runtime") or {}
    if runtime != {
        "isaaclab_launcher": "/home/pacquadr/IsaacLab/isaaclab.sh",
        "visualization": "none",
        "simulation_device": "cpu",
        "policy_device": "cuda",
        "observation_preset": "core_door_pose",
        "wandb_mode": "disabled",
    }:
        raise UnifiedEvalError("runtime contract drifted")
    if {
        key: rollout.get(key)
        for key in (
            "success_angle_deg",
            "max_ticks",
            "success_semantics",
            "adapter_version",
            "contact_entry_shaping",
            "force_admission_watch_bound_n",
            "warning_adjudication",
        )
    } != {
        "success_angle_deg": 45.0,
        "max_ticks": 600,
        "success_semantics": "per_tick_first_crossing_stop",
        "adapter_version": "adapter-v1",
        "contact_entry_shaping": True,
        "force_admission_watch_bound_n": 200.0,
        "warning_adjudication": "alexdoor.warning-adjudication.v3",
    }:
        raise UnifiedEvalError("rollout/safety contract drifted")
    if raw.get("policies") != {
        "act": {
            "checkpoint_horizon": 40,
            "temporal_ensemble": False,
            "ensemble_m": 0.01,
            "execution_mode": "chunk_execution",
        },
        "diffusion": {
            "sampler": "ddim",
            "num_inference_steps": 10,
            "checkpoint_horizon": 16,
            "model_horizon": 16,
            "n_action_steps": 8,
            "execution_mode": "receding_horizon",
        },
    }:
        raise UnifiedEvalError("policy inference contract drifted")

    inventory_hashes = _resolve(repo, raw["immutable_inventory"])
    inventory_sizes = inventory_hashes.with_name("returned_package_inventory.tsv")
    return UnifiedPlan(
        path=plan_path,
        repo_root=repo,
        raw=raw,
        protocol_id=raw["protocol_id"],
        source_git_commit=raw["source_git_commit"],
        attempt_id=raw["attempt_id"],
        transfer_manifest=_resolve(repo, raw["transfer_manifest"]),
        return_root=return_root,
        workspace_root=workspace_root,
        curated_root=curated_root,
        inventory_hashes=inventory_hashes,
        inventory_sizes=inventory_sizes,
        poses=poses,
        cells=cells,
    )


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise UnifiedEvalError(f"{label} may not be a symlink: {path}")


def _copy_exact(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    _reject_symlink(source, "source artifact")
    if not source.is_file():
        raise UnifiedEvalError(f"source artifact is missing: {source}")
    expected = expected_sha256 or sha256_file(source)
    if destination.exists() or destination.is_symlink():
        _reject_symlink(destination, "workspace artifact")
        if not destination.is_file() or sha256_file(destination) != expected:
            raise UnifiedEvalError(f"conflicting workspace artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)
    if sha256_file(destination) != expected:
        raise UnifiedEvalError(f"workspace copy hash mismatch: {destination}")


def _publish_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() or path.is_symlink():
        _reject_symlink(path, "published artifact")
        if path.read_text() == content:
            return
        raise UnifiedEvalError(f"refusing to overwrite conflicting artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise UnifiedEvalError(f"concurrent publication conflict: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def verify_immutable_inventory(plan: UnifiedPlan) -> list[str]:
    failures: list[str] = []
    if not plan.inventory_hashes.is_file() or not plan.inventory_sizes.is_file():
        return ["immutable returned-package baselines are missing"]
    expected_sizes: dict[str, int] = {}
    for line in plan.inventory_sizes.read_text().splitlines():
        try:
            relative, size = line.split("\t")
            expected_sizes[relative] = int(size)
        except (ValueError, TypeError):
            failures.append(f"malformed size inventory row: {line!r}")
    expected_hashes: dict[str, str] = {}
    for line in plan.inventory_hashes.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected_hashes[relative.removeprefix("./")] = digest
    actual_paths = {
        path.relative_to(plan.return_root).as_posix(): path
        for path in plan.return_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    symlinks = [path for path in plan.return_root.rglob("*") if path.is_symlink()]
    if symlinks:
        failures.append(f"returned package contains symlinks: {symlinks}")
    if set(actual_paths) != set(expected_sizes) or set(actual_paths) != set(expected_hashes):
        failures.append("returned package exact path inventory changed")
        return failures
    for relative, path in actual_paths.items():
        if path.stat().st_size != expected_sizes[relative]:
            failures.append(f"returned size changed: {relative}")
        if sha256_file(path) != expected_hashes[relative]:
            failures.append(f"returned hash changed: {relative}")
    return failures


def prepare_workspace(plan: UnifiedPlan) -> dict[str, Any]:
    return_failures = verify_immutable_inventory(plan)
    if return_failures:
        raise UnifiedEvalError("; ".join(return_failures))
    resolved_cells: list[dict[str, Any]] = []
    for cell in plan.cells:
        source_run = cell.returned_checkpoint.parents[1]
        workspace_run = cell.workspace_checkpoint.parents[1]
        _copy_exact(cell.returned_checkpoint, cell.workspace_checkpoint, cell.checkpoint_sha256)
        for relative in (
            Path("logs/train_log.json"),
            Path("resolved_config.json"),
            Path("metrics/open_loop.json"),
        ):
            _copy_exact(source_run / relative, workspace_run / relative)
        train_log = json.loads((workspace_run / "logs/train_log.json").read_text())
        training_provenance = train_log.get("training_provenance") or {}
        required_training = {
            "master_dataset_fingerprint_sha256",
            "action_dataset_fingerprint_sha256",
            "view_id",
            "view_fingerprint_sha256",
            "split_fingerprint_sha256",
            "split_episode_ids",
            "normalization_sha256",
            "normalization_fingerprint_sha256",
            "source_git_commit",
            "resolved_training_config_sha256",
        }
        missing = sorted(required_training - set(training_provenance))
        if missing:
            raise UnifiedEvalError(f"{cell.run_id} training provenance missing {missing}")
        if training_provenance["source_git_commit"] != plan.source_git_commit:
            raise UnifiedEvalError(f"{cell.run_id} training source commit mismatch")
        resolved_cells.append(
            {
                **{
                    key: value
                    for key, value in plan.raw["cells"][cell.index].items()
                },
                "returned_checkpoint": str(cell.returned_checkpoint),
                "workspace_checkpoint": str(cell.workspace_checkpoint),
                "training_provenance": training_provenance,
            }
        )
    resolved = {
        "schema": "alexdoor_xas.phase3_unified_resolved_plan.v1",
        "protocol_id": plan.protocol_id,
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "source_git_commit": plan.source_git_commit,
        "attempt_id": plan.attempt_id,
        "return_root": str(plan.return_root),
        "workspace_root": str(plan.workspace_root),
        "curated_root": str(plan.curated_root),
        "immutable_inventory": {
            "hashes": str(plan.inventory_hashes),
            "sizes": str(plan.inventory_sizes),
            "file_count": len(plan.inventory_hashes.read_text().splitlines()),
        },
        "code_provenance": code_provenance(plan),
        "rollout": plan.raw["rollout"],
        "runtime": plan.raw["runtime"],
        "policies": plan.raw["policies"],
        "statistics": plan.raw["statistics"],
        "cells": resolved_cells,
    }
    output = plan.workspace_root / "provenance" / "evaluation_plan.resolved.json"
    _publish_exclusive_json(output, resolved)
    return resolved


def code_provenance(plan: UnifiedPlan) -> dict[str, str]:
    files = {
        "driver": plan.repo_root / "scripts" / "run_phase3_unified_evaluation.py",
        "helper": Path(__file__).resolve(),
        "act_evaluator": plan.repo_root / "scripts" / "eval_act.py",
        "diffusion_evaluator": plan.repo_root / "scripts" / "eval_diffusion.py",
        "plan": plan.path,
    }
    return {
        f"{name}_path": str(path)
        for name, path in files.items()
    } | {
        f"{name}_sha256": sha256_file(path) if path.is_file() else "MISSING"
        for name, path in files.items()
    }


def _next_execution_root(plan: UnifiedPlan, stage: str, cell: CellPlan, pose: PosePlan) -> Path:
    parent = plan.workspace_root / "executions" / stage / cell.run_id / pose.pose_id
    parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        candidate = parent / f"attempt{attempt:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise UnifiedEvalError(f"too many execution attempts under {parent}")


def _stage_run(plan: UnifiedPlan, execution_root: Path, cell: CellPlan) -> Path:
    source_run = cell.workspace_checkpoint.parents[1]
    stage_run = execution_root / "run"
    for relative in (
        Path("checkpoints/best.pt"),
        Path("logs/train_log.json"),
        Path("resolved_config.json"),
        Path("metrics/open_loop.json"),
    ):
        expected = cell.checkpoint_sha256 if relative == Path("checkpoints/best.pt") else None
        _copy_exact(source_run / relative, stage_run / relative, expected)
    if stage_run.is_relative_to(plan.return_root):
        raise UnifiedEvalError("staged run unexpectedly resolves inside the return package")
    return stage_run


def build_eval_command(
    plan: UnifiedPlan,
    cell: CellPlan,
    pose: PosePlan,
    checkpoint: Path,
    *,
    episodes_fixed: int | None = None,
    episodes_randomized: int | None = None,
) -> list[str]:
    fixed = pose.episodes_fixed if episodes_fixed is None else episodes_fixed
    randomized = pose.episodes_randomized if episodes_randomized is None else episodes_randomized
    script = "scripts/eval_act.py" if cell.policy == "act" else "scripts/eval_diffusion.py"
    command = [
        plan.raw["runtime"]["isaaclab_launcher"],
        "-p",
        script,
        "--viz",
        plan.raw["runtime"]["visualization"],
        "--device",
        plan.raw["runtime"]["simulation_device"],
        "--checkpoint",
        str(checkpoint),
    ]
    if cell.policy == "diffusion":
        command.extend(["--sampler", "ddim", "--inference-steps", "10"])
    command.extend(
        [
            "model.chunk_size=40" if cell.policy == "act" else "model.horizon=16",
            "rollout.policy_device=cuda",
            f"rollout.episodes_fixed={fixed}",
            f"rollout.episodes_randomized={randomized}",
            f"rollout.base_seed={pose.base_seed}",
            "rollout.max_ticks=600",
            "rollout.success_angle_deg=45",
            f"rollout.door_pose_id={pose.pose_id}",
            f"rollout.door_yaw_deg={pose.door_yaw_deg}",
            f"rollout.door_offset_x={pose.door_offset_x_m}",
            f"rollout.door_offset_y={pose.door_offset_y_m}",
        ]
    )
    if cell.policy == "act":
        command.extend(["rollout.temporal_ensemble=false", "rollout.ensemble_m=0.01"])
    else:
        command.append("rollout.n_action_steps=8")
    command.append("+wandb.mode=disabled")
    return command


def canonical_pose_path(plan: UnifiedPlan, stage: str, cell: CellPlan, pose: PosePlan) -> Path:
    return plan.workspace_root / "results" / stage / cell.run_id / f"{pose.pose_id}.json"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnifiedEvalError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise UnifiedEvalError(f"{label} must be finite")
    return result


def _same_float(actual: Any, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(_finite(actual, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise UnifiedEvalError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_force(row: dict[str, Any], label: str) -> None:
    if row["contact_source"] != "force_sensor" or row["contact_unavailable_reason"] is not None:
        raise UnifiedEvalError(f"{label} lacks required force-sensor contact evidence")
    force = row["force_n"]
    all_samples = row["force_n_all_samples"]
    trace = row["force_trace_evidence"]
    if (
        not isinstance(force, dict)
        or not isinstance(all_samples, dict)
        or not isinstance(trace, dict)
    ):
        raise UnifiedEvalError(f"{label} lacks required force summaries/trace")
    mean = _finite(force.get("mean"), f"{label} force mean")
    maximum = _finite(force.get("max"), f"{label} force max")
    p95 = _finite(force.get("p95"), f"{label} force p95")
    impulse = _finite(row["impulse_ns"], f"{label} force impulse")
    all_max = _finite(all_samples.get("max"), f"{label} all-sample force max")
    if min(mean, maximum, p95, impulse, all_max) < 0 or mean > maximum or p95 > maximum:
        raise UnifiedEvalError(f"{label} has inconsistent nonnegative force evidence")
    count = all_samples.get("n_exceedance_ticks")
    if not _is_int(count) or count < 0:
        raise UnifiedEvalError(f"{label} has invalid force exceedance count")
    if row["force_exceeds_admission_bound"] is not (count > 0):
        raise UnifiedEvalError(f"{label} force exceedance boolean is inconsistent")
    _same_float(trace.get("admission_bound_n"), 200.0, f"{label} force trace bound")
    _same_float(trace.get("peak_force_n"), all_max, f"{label} force trace peak", 1e-6)
    if trace.get("n_exceedance_ticks") != count:
        raise UnifiedEvalError(f"{label} force trace exceedance count mismatch")
    ticks = trace.get("exceedance_ticks")
    if not isinstance(ticks, list) or len(ticks) != count:
        raise UnifiedEvalError(f"{label} force exceedance ticks mismatch")
    _require_sha256(trace.get("trace_sha256"), f"{label} force trace")


def validate_eval_payload(
    plan: UnifiedPlan,
    cell: CellPlan,
    pose: PosePlan,
    payload: dict[str, Any],
    *,
    expected_fixed: int | None = None,
    expected_randomized: int | None = None,
) -> list[dict[str, Any]]:
    fixed = pose.episodes_fixed if expected_fixed is None else expected_fixed
    randomized = pose.episodes_randomized if expected_randomized is None else expected_randomized
    expected_count = fixed + randomized
    for key, expected in (
        ("policy", cell.policy),
        ("action_space", cell.action_space),
        ("obs_preset", "core_door_pose"),
        ("checkpoint_sha256", cell.checkpoint_sha256),
        ("policy_device", "cuda"),
        ("max_ticks", 600),
        ("success_semantics", "per_tick_first_crossing_stop"),
        ("base_seed", pose.base_seed),
    ):
        if payload.get(key) != expected:
            raise UnifiedEvalError(f"{cell.run_id}/{pose.pose_id} top-level {key} mismatch")
    _same_float(payload.get("success_angle_deg"), 45.0, "success angle")
    checkpoint = Path(str(payload.get("checkpoint"))).resolve()
    if checkpoint.is_relative_to(plan.return_root) or not checkpoint.is_relative_to(
        plan.workspace_root
    ):
        raise UnifiedEvalError("evaluation checkpoint was not staged outside the return package")
    door_pose = payload.get("door_pose") or {}
    if door_pose.get("door_pose_id") != pose.pose_id:
        raise UnifiedEvalError("door pose ID mismatch")
    _same_float(door_pose.get("door_yaw_deg"), pose.door_yaw_deg, "door yaw", 1e-6)
    offsets = door_pose.get("door_offset_xy")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise UnifiedEvalError("door offset geometry is missing")
    _same_float(offsets[0], pose.door_offset_x_m, "door offset x")
    _same_float(offsets[1], pose.door_offset_y_m, "door offset y")
    protocol = payload.get("seed_protocol") or {}
    expected_fixed_seeds = list(range(pose.base_seed, pose.base_seed + fixed))
    random_start = pose.base_seed + fixed
    expected_randomized_seeds = list(range(random_start, random_start + randomized))
    if {
        key: protocol.get(key)
        for key in (
            "base_seed",
            "episodes_fixed",
            "episodes_randomized",
            "fixed_seeds",
            "randomized_seeds",
        )
    } != {
        "base_seed": pose.base_seed,
        "episodes_fixed": fixed,
        "episodes_randomized": randomized,
        "fixed_seeds": expected_fixed_seeds,
        "randomized_seeds": expected_randomized_seeds,
    }:
        raise UnifiedEvalError("seed protocol mismatch")
    if cell.policy == "act":
        expected_policy = {
            "chunk_size": 40,
            "checkpoint_horizon": 40,
            "temporal_ensemble": False,
            "ensemble_m": 0.01,
            "execution_mode": "chunk_execution",
        }
    else:
        expected_policy = {
            "horizon": 16,
            "checkpoint_horizon": 16,
            "n_action_steps": 8,
            "sampler": "ddim",
            "num_inference_steps": 10,
            "execution_mode": "receding_horizon",
        }
    for key, expected in expected_policy.items():
        if payload.get(key) != expected:
            raise UnifiedEvalError(f"policy metadata mismatch on {key}")

    provenance = payload.get("dataset_provenance") or {}
    missing_provenance = sorted(REQUIRED_PROVENANCE_FIELDS - set(provenance))
    if missing_provenance:
        raise UnifiedEvalError(f"dataset provenance missing {missing_provenance}")
    dataset = provenance.get("dataset") or {}
    expected_dataset = {
        "task": "door_push_alex_v2",
        "space": cell.action_space,
        "version": "v3_scale_master",
        "view_id": cell.view_id,
        "obs_preset": "core_door_pose",
    }
    if {key: dataset.get(key) for key in expected_dataset} != expected_dataset:
        raise UnifiedEvalError("checkpoint dataset identity mismatch")
    for key in (
        "dataset_fingerprint_match",
        "split_fingerprint_match",
        "train_split_match",
        "val_split_match",
        "val_split_checked",
    ):
        if provenance.get(key) is not True:
            raise UnifiedEvalError(f"dataset provenance gate failed: {key}")
    for key in REQUIRED_PROVENANCE_FIELDS - {"split_episode_ids", "view_id"}:
        _require_sha256(provenance.get(key), f"dataset provenance {key}")
    split_ids = provenance.get("split_episode_ids") or {}
    if set(split_ids) != {"train", "val", "test"}:
        raise UnifiedEvalError("split membership is incomplete")
    training = json.loads(
        (cell.workspace_checkpoint.parents[1] / "logs/train_log.json").read_text()
    ).get("training_provenance") or {}
    for key in (
        "master_dataset_fingerprint_sha256",
        "action_dataset_fingerprint_sha256",
        "view_id",
        "view_fingerprint_sha256",
        "split_fingerprint_sha256",
        "normalization_sha256",
    ):
        if provenance.get(key) != training.get(key):
            raise UnifiedEvalError(f"evaluation/training provenance mismatch: {key}")

    rows = payload.get("rollouts")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise UnifiedEvalError(f"expected {expected_count} rollout rows")
    expected_seed_types = [(seed, False) for seed in expected_fixed_seeds] + [
        (seed, True) for seed in expected_randomized_seeds
    ]
    actual_seed_types: list[tuple[int, bool]] = []
    control_dt = _finite(payload.get("control_dt"), "control_dt")
    for index, row in enumerate(rows):
        label = f"{cell.run_id}/{pose.pose_id}/row{index}"
        if not isinstance(row, dict):
            raise UnifiedEvalError(f"{label} is not an object")
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise UnifiedEvalError(f"{label} missing fields {missing}")
        if not _is_int(row["seed"]) or not isinstance(row["randomized"], bool):
            raise UnifiedEvalError(f"{label} has invalid seed/type")
        actual_seed_types.append((row["seed"], row["randomized"]))
        if row["door_pose_id"] != pose.pose_id:
            raise UnifiedEvalError(f"{label} pose ID mismatch")
        _same_float(row["door_yaw_deg"], pose.door_yaw_deg, f"{label} yaw", 1e-6)
        row_offsets = row["door_offset_xy"]
        if not isinstance(row_offsets, list) or len(row_offsets) != 2:
            raise UnifiedEvalError(f"{label} offset is invalid")
        _same_float(row_offsets[0], pose.door_offset_x_m, f"{label} offset x")
        _same_float(row_offsets[1], pose.door_offset_y_m, f"{label} offset y")
        if not isinstance(row["success"], bool) or not isinstance(row["env_truncated"], bool):
            raise UnifiedEvalError(f"{label} has invalid success/truncation flags")
        for numeric in ("initial_angle_rad", "final_angle_rad", "door_angle_change_rad"):
            _finite(row[numeric], f"{label} {numeric}")
        if not _is_int(row["n_ticks"]) or not 1 <= row["n_ticks"] <= 600:
            raise UnifiedEvalError(f"{label} tick count is invalid")
        for count_key in (
            "n_accepted",
            "n_corrected",
            "n_rejected",
            "n_warnings",
            "contact_ticks",
        ):
            if not _is_int(row[count_key]) or row[count_key] < 0:
                raise UnifiedEvalError(f"{label} {count_key} is invalid")
        if row["n_accepted"] + row["n_corrected"] + row["n_rejected"] != row["n_ticks"]:
            raise UnifiedEvalError(f"{label} adapter decision counts do not cover every tick")
        if row["contact_ticks"] > row["n_ticks"]:
            raise UnifiedEvalError(f"{label} contact ticks exceed total ticks")
        if row["success"] != (row["failure_label"] is None):
            raise UnifiedEvalError(f"{label} success/failure label mismatch")
        if row["success"]:
            if row["termination_reason"] != "success" or not _is_int(
                row["first_success_tick"]
            ):
                raise UnifiedEvalError(f"{label} success termination metadata is invalid")
            if not 1 <= row["first_success_tick"] <= row["n_ticks"]:
                raise UnifiedEvalError(f"{label} first success tick is invalid")
            _same_float(
                row["time_to_success_s"],
                row["first_success_tick"] * control_dt,
                f"{label} time to success",
                1e-7,
            )
        elif row["first_success_tick"] is not None or row["time_to_success_s"] is not None:
            raise UnifiedEvalError(f"{label} failed rollout has success timing")
        warnings = row["warning_records"]
        if not isinstance(warnings, list) or len(warnings) != row["n_warnings"]:
            raise UnifiedEvalError(f"{label} warning records are incomplete")
        for warning in warnings:
            if not isinstance(warning, dict) or not {"id", "message", "evidence"}.issubset(
                warning
            ):
                raise UnifiedEvalError(f"{label} warning record schema is invalid")
        _validate_force(row, label)
        metadata_keys = row["policy_metadata_keys"]
        if not isinstance(metadata_keys, list) or not set(expected_policy).issubset(metadata_keys):
            raise UnifiedEvalError(f"{label} policy metadata declaration is incomplete")
        for key, expected in expected_policy.items():
            if row.get(key) != expected:
                raise UnifiedEvalError(f"{label} policy metadata mismatch on {key}")
    if actual_seed_types != expected_seed_types or len(set(actual_seed_types)) != len(
        actual_seed_types
    ):
        raise UnifiedEvalError("rollout rows do not match the ordered unique seed plan")
    aggregate = payload.get("aggregate") or {}
    if aggregate.get("n_rollouts") != expected_count or aggregate.get("n_success") != sum(
        bool(row["success"]) for row in rows
    ):
        raise UnifiedEvalError("aggregate rollout counts disagree with rows")
    return rows


def _execution_provenance(
    plan: UnifiedPlan,
    cell: CellPlan,
    pose: PosePlan,
    stage: str,
    execution_root: Path,
    checkpoint: Path,
    command: list[str],
) -> dict[str, Any]:
    train_log = json.loads((checkpoint.parents[1] / "logs/train_log.json").read_text())
    return {
        "protocol_id": plan.protocol_id,
        "primary_or_diagnostic": "primary" if stage == "primary" else "preflight",
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "source_git_commit": plan.source_git_commit,
        "attempt_id": plan.attempt_id,
        "cell_index": cell.index,
        "run_id": cell.run_id,
        "dataset_size": cell.dataset_size,
        "view_id": cell.view_id,
        "returned_checkpoint": str(cell.returned_checkpoint),
        "workspace_checkpoint": str(cell.workspace_checkpoint),
        "staged_checkpoint": str(checkpoint),
        "checkpoint_sha256": cell.checkpoint_sha256,
        "training_provenance": train_log["training_provenance"],
        "simulation_device": "cpu",
        "policy_device": "cuda",
        "adapter_version": "adapter-v1",
        "contact_entry_shaping": True,
        "force_admission_watch_bound_n": 200.0,
        "pose_id": pose.pose_id,
        "argv": command,
        "execution_root": str(execution_root),
        "stdout_log": str(execution_root / "stdout.log"),
        "stderr_log": str(execution_root / "stderr.log"),
        "code": code_provenance(plan),
    }


def run_pose(
    plan: UnifiedPlan,
    cell: CellPlan,
    pose: PosePlan,
    *,
    stage: str,
    episodes_fixed: int | None = None,
    episodes_randomized: int | None = None,
) -> Path:
    if stage not in {"primary", "preflight"}:
        raise UnifiedEvalError(f"unsupported evaluation stage {stage!r}")
    canonical = canonical_pose_path(plan, stage, cell, pose)
    if canonical.exists():
        payload = json.loads(canonical.read_text())
        validate_eval_payload(
            plan,
            cell,
            pose,
            payload,
            expected_fixed=episodes_fixed,
            expected_randomized=episodes_randomized,
        )
        evidence = payload.get("evaluation_provenance") or {}
        if evidence.get("plan_sha256") != plan.sha256 or evidence.get("run_id") != cell.run_id:
            raise UnifiedEvalError(f"canonical provenance mismatch: {canonical}")
        return canonical
    execution_root = _next_execution_root(plan, stage, cell, pose)
    staged_run = _stage_run(plan, execution_root, cell)
    command = build_eval_command(
        plan,
        cell,
        pose,
        staged_run / "checkpoints/best.pt",
        episodes_fixed=episodes_fixed,
        episodes_randomized=episodes_randomized,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(plan.repo_root)
    with (execution_root / "stdout.log").open("x") as stdout, (
        execution_root / "stderr.log"
    ).open("x") as stderr:
        result = subprocess.run(
            command,
            cwd=plan.repo_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise UnifiedEvalError(
            f"{cell.run_id}/{pose.pose_id} evaluator failed with exit {result.returncode}; "
            f"see {execution_root}"
        )
    name = f"act_eval_{pose.pose_id}.json" if cell.policy == "act" else (
        f"diffusion_eval_{pose.pose_id}.json"
    )
    raw_path = staged_run / "metrics" / name
    try:
        payload = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UnifiedEvalError(f"invalid evaluator output {raw_path}: {error}") from error
    validate_eval_payload(
        plan,
        cell,
        pose,
        payload,
        expected_fixed=episodes_fixed,
        expected_randomized=episodes_randomized,
    )
    payload["schema"] = CANONICAL_SCHEMA
    payload["evaluation_provenance"] = _execution_provenance(
        plan,
        cell,
        pose,
        stage,
        execution_root,
        staged_run / "checkpoints/best.pt",
        command,
    )
    _publish_exclusive_json(canonical, payload)
    validate_eval_payload(
        plan,
        cell,
        pose,
        json.loads(canonical.read_text()),
        expected_fixed=episodes_fixed,
        expected_randomized=episodes_randomized,
    )
    return canonical


def run_preflight(plan: UnifiedPlan, run_id: str | None = None) -> list[Path]:
    configured = tuple(plan.raw["preflight"]["cells"])
    selected = configured if run_id is None else (run_id,)
    if any(value not in configured for value in selected):
        raise UnifiedEvalError("preflight is limited to the frozen ACT-A2 and Diffusion-A3 cells")
    pose = plan.pose(plan.raw["preflight"]["pose_id"])
    return [
        run_pose(
            plan,
            plan.cell(value),
            pose,
            stage="preflight",
            episodes_fixed=1,
            episodes_randomized=0,
        )
        for value in selected
    ]


def _completion_payload(
    plan: UnifiedPlan, cell: CellPlan, pose_paths: list[Path]
) -> dict[str, Any]:
    pose_artifacts = {
        path.stem: {"path": str(path), "sha256": sha256_file(path)} for path in pose_paths
    }
    return {
        "schema": COMPLETION_SCHEMA,
        "protocol_id": plan.protocol_id,
        "plan_sha256": plan.sha256,
        "source_git_commit": plan.source_git_commit,
        "attempt_id": plan.attempt_id,
        "cell_index": cell.index,
        "run_id": cell.run_id,
        "policy": cell.policy,
        "action_space": cell.action_space,
        "dataset_size": cell.dataset_size,
        "view_id": cell.view_id,
        "checkpoint_sha256": cell.checkpoint_sha256,
        "expected_rollouts": 36,
        "pose_artifacts": pose_artifacts,
        "status": "COMPLETED",
    }


def run_cell(plan: UnifiedPlan, run_id: str) -> Path:
    cell = plan.cell(run_id)
    pose_paths = [run_pose(plan, cell, pose, stage="primary") for pose in plan.poses]
    all_rows: list[dict[str, Any]] = []
    for pose, path in zip(plan.poses, pose_paths, strict=True):
        all_rows.extend(validate_eval_payload(plan, cell, pose, json.loads(path.read_text())))
    keys = [(row["door_pose_id"], row["seed"], row["randomized"]) for row in all_rows]
    if len(all_rows) != 36 or len(set(keys)) != 36:
        raise UnifiedEvalError(f"{cell.run_id} does not contain 36 unique primary rollouts")
    completion = plan.workspace_root / "results" / "primary" / cell.run_id / "completion.json"
    _publish_exclusive_json(completion, _completion_payload(plan, cell, pose_paths))
    return completion


def _load_primary_cell(plan: UnifiedPlan, cell: CellPlan) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for pose in plan.poses:
        path = canonical_pose_path(plan, "primary", cell, pose)
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing canonical pose artifact {pose.pose_id}")
            continue
        try:
            payload = json.loads(path.read_text())
            pose_rows = validate_eval_payload(plan, cell, pose, payload)
            evidence = payload.get("evaluation_provenance") or {}
            expected = {
                "protocol_id": plan.protocol_id,
                "primary_or_diagnostic": "primary",
                "plan_sha256": plan.sha256,
                "source_git_commit": plan.source_git_commit,
                "attempt_id": plan.attempt_id,
                "cell_index": cell.index,
                "run_id": cell.run_id,
                "dataset_size": cell.dataset_size,
                "view_id": cell.view_id,
                "checkpoint_sha256": cell.checkpoint_sha256,
                "simulation_device": "cpu",
                "policy_device": "cuda",
                "adapter_version": "adapter-v1",
                "contact_entry_shaping": True,
                "force_admission_watch_bound_n": 200.0,
                "pose_id": pose.pose_id,
            }
            for key, value in expected.items():
                if evidence.get(key) != value:
                    raise UnifiedEvalError(f"evaluation provenance mismatch on {key}")
            training = evidence.get("training_provenance") or {}
            if training.get("normalization_fingerprint_sha256") is None:
                raise UnifiedEvalError("normalization semantic fingerprint is missing")
            rows.extend(pose_rows)
        except (OSError, json.JSONDecodeError, UnifiedEvalError) as error:
            problems.append(f"{pose.pose_id}: {error}")
    keys = [(row.get("door_pose_id"), row.get("seed"), row.get("randomized")) for row in rows]
    if len(rows) != 36:
        problems.append(f"expected 36 primary rollouts, found {len(rows)}")
    if len(set(keys)) != len(keys):
        problems.append("duplicate matched rollout keys")
    completion = plan.workspace_root / "results" / "primary" / cell.run_id / "completion.json"
    if not completion.is_file():
        problems.append("cell completion record is missing")
    else:
        try:
            recorded = json.loads(completion.read_text())
            expected_completion = _completion_payload(
                plan,
                cell,
                [canonical_pose_path(plan, "primary", cell, pose) for pose in plan.poses],
            )
            if recorded != expected_completion:
                problems.append("cell completion record is stale or inconsistent")
        except (OSError, json.JSONDecodeError, FileNotFoundError) as error:
            problems.append(f"invalid cell completion record: {error}")
    return rows, problems


def _artifact_row(plan: UnifiedPlan, cell: CellPlan, status: str, reasons: list[str]) -> dict:
    returned_run = cell.returned_checkpoint.parents[1]
    workspace_run = cell.workspace_checkpoint.parents[1]
    train_log = json.loads((workspace_run / "logs/train_log.json").read_text())
    training = train_log["training_provenance"]
    return_manifest = json.loads(
        (
            plan.return_root
            / ".sweep_return"
            / "attempts"
            / plan.attempt_id
            / "return_manifest.json"
        ).read_text()
    )
    wandb_paths = sorted(
        str(path) for path in (returned_run / "wandb").rglob("offline-run-*") if path.is_dir()
    )
    eval_paths = [
        str(canonical_pose_path(plan, "primary", cell, pose)) for pose in plan.poses
    ]
    return {
        "policy": cell.policy,
        "action_space": cell.action_space,
        "dataset_size": cell.dataset_size,
        "view_id": cell.view_id,
        "run_id": cell.run_id,
        "attempt_id": plan.attempt_id,
        "source_git_commit": plan.source_git_commit,
        "master_dataset_fingerprint": training["master_dataset_fingerprint_sha256"],
        "action_dataset_fingerprint": training["action_dataset_fingerprint_sha256"],
        "view_fingerprint": training["view_fingerprint_sha256"],
        "split_id": training["split_fingerprint_sha256"],
        "norm_stats_id": training["normalization_sha256"],
        "normalization_fingerprint": training["normalization_fingerprint_sha256"],
        "train_run_id": train_log["run_id"],
        "checkpoint_path": str(cell.returned_checkpoint),
        "checkpoint_sha256": cell.checkpoint_sha256,
        "open_loop_report_path": str(returned_run / "metrics/open_loop.json"),
        "closed_loop_eval_json_paths": json.dumps(eval_paths, separators=(",", ":")),
        "per_rollout_table_path": str(plan.curated_root / "normalized_rollouts.csv"),
        "wandb_or_offline_log_path": json.dumps(wandb_paths, separators=(",", ":")),
        "return_manifest_file_count": return_manifest["file_count"],
        "primary_or_diagnostic": "primary",
        "status": status,
        "exclusion_reason": "; ".join(reasons),
    }


def audit_evidence(plan: UnifiedPlan) -> dict[str, Any]:
    immutable_failures = verify_immutable_inventory(plan)
    if immutable_failures:
        raise UnifiedEvalError("immutable return audit failed: " + "; ".join(immutable_failures))
    cells: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    included_rows: dict[str, list[dict[str, Any]]] = {}
    for cell in plan.cells:
        rows, problems = _load_primary_cell(plan, cell)
        pose_files = sum(
            canonical_pose_path(plan, "primary", cell, pose).is_file() for pose in plan.poses
        )
        if problems:
            status = "partial" if 0 < pose_files < 5 else "excluded"
            exclusions.append(
                {
                    "run_id": cell.run_id,
                    "policy": cell.policy,
                    "action_space": cell.action_space,
                    "dataset_size": cell.dataset_size,
                    "reasons": problems,
                }
            )
        else:
            status = "complete"
            included_rows[cell.run_id] = rows
        cells.append(_artifact_row(plan, cell, status, problems))
    return {
        "schema": "alexdoor_xas.phase3_unified_audit.v1",
        "protocol_id": plan.protocol_id,
        "plan_sha256": plan.sha256,
        "attempt_id": plan.attempt_id,
        "source_git_commit": plan.source_git_commit,
        "cells": cells,
        "exclusions": exclusions,
        "included_run_ids": sorted(included_rows),
        "completed_cells": len(included_rows),
        "excluded_cells": len(plan.cells) - len(included_rows),
        "primary_rollouts": sum(len(rows) for rows in included_rows.values()),
        "_rows": included_rows,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_rows(
    plan: UnifiedPlan, included: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for cell in plan.cells:
        if cell.run_id not in included:
            continue
        training = json.loads(
            (cell.workspace_checkpoint.parents[1] / "logs/train_log.json").read_text()
        )["training_provenance"]
        for row in included[cell.run_id]:
            force = row["force_n"]
            all_samples = row["force_n_all_samples"]
            trace = row["force_trace_evidence"]
            normalized.append(
                {
                    "policy": cell.policy,
                    "action_space": cell.action_space,
                    "dataset_size": cell.dataset_size,
                    "run_id": cell.run_id,
                    "eval_protocol_id": plan.protocol_id,
                    "attempt_id": plan.attempt_id,
                    "source_git_commit": plan.source_git_commit,
                    "checkpoint_path": str(cell.returned_checkpoint),
                    "checkpoint_sha256": cell.checkpoint_sha256,
                    "master_dataset_fingerprint": training[
                        "master_dataset_fingerprint_sha256"
                    ],
                    "action_dataset_fingerprint": training[
                        "action_dataset_fingerprint_sha256"
                    ],
                    "view_id": cell.view_id,
                    "view_fingerprint": training["view_fingerprint_sha256"],
                    "split_id": training["split_fingerprint_sha256"],
                    "normalization_sha256": training["normalization_sha256"],
                    "normalization_fingerprint": training[
                        "normalization_fingerprint_sha256"
                    ],
                    "obs_preset": "core_door_pose",
                    "adapter_version": "adapter-v1",
                    "success_angle_deg": 45.0,
                    "success_semantics": "per_tick_first_crossing_stop",
                    "max_ticks": 600,
                    "control_dt": 1.0 / 60.0,
                    "seed": row["seed"],
                    "fixed_or_randomized": "randomized" if row["randomized"] else "fixed",
                    "door_pose_id": row["door_pose_id"],
                    "door_orientation_id": row["door_pose_id"],
                    "door_yaw_deg": row["door_yaw_deg"],
                    "door_offset_x_m": row["door_offset_xy"][0],
                    "door_offset_y_m": row["door_offset_xy"][1],
                    "success": row["success"],
                    "initial_angle_rad": row["initial_angle_rad"],
                    "final_angle_rad": row["final_angle_rad"],
                    "final_angle_deg": math.degrees(row["final_angle_rad"]),
                    "door_angle_change_rad": row["door_angle_change_rad"],
                    "total_ticks": row["n_ticks"],
                    "ticks_to_success": row["first_success_tick"],
                    "time_to_success_s": row["time_to_success_s"],
                    "termination_reason": row["termination_reason"],
                    "failure_label": row["failure_label"],
                    "env_truncated": row["env_truncated"],
                    "adapter_accepted": row["n_accepted"],
                    "adapter_corrected": row["n_corrected"],
                    "adapter_rejected": row["n_rejected"],
                    "adapter_warning_count": row["n_warnings"],
                    "adapter_warning_types": _canonical_json(row["warning_family_counts"]),
                    "adapter_warning_records": _canonical_json(row["warning_records"]),
                    "contact_ticks": row["contact_ticks"],
                    "contact_source": row["contact_source"],
                    "contact_unavailable_reason": row["contact_unavailable_reason"],
                    "mean_contact_force_n": force["mean"],
                    "max_contact_force_n": force["max"],
                    "p95_contact_force_n": force["p95"],
                    "contact_force_impulse_ns": row["impulse_ns"],
                    "all_sample_max_force_n": all_samples["max"],
                    "force_exceeds_admission_bound": row[
                        "force_exceeds_admission_bound"
                    ],
                    "force_exceedance_ticks": all_samples["n_exceedance_ticks"],
                    "force_trace_sha256": trace["trace_sha256"],
                    "act_chunk_size": row.get("chunk_size"),
                    "act_checkpoint_horizon": (
                        row.get("checkpoint_horizon") if cell.policy == "act" else None
                    ),
                    "act_temporal_ensemble": row.get("temporal_ensemble"),
                    "diffusion_sampler": row.get("sampler"),
                    "diffusion_inference_steps": row.get("num_inference_steps"),
                    "diffusion_model_horizon": row.get("horizon"),
                    "diffusion_checkpoint_horizon": (
                        row.get("checkpoint_horizon")
                        if cell.policy == "diffusion"
                        else None
                    ),
                    "diffusion_action_horizon": row.get("n_action_steps"),
                    "primary_or_diagnostic": "primary",
                    "source_eval_json": str(
                        canonical_pose_path(plan, "primary", cell, plan.pose(row["door_pose_id"]))
                    ),
                    "notes": row["notes"],
                }
            )
    normalized.sort(
        key=lambda row: (
            row["dataset_size"],
            row["policy"],
            row["action_space"],
            row["door_pose_id"],
            row["seed"],
        )
    )
    return normalized


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    if confidence != 0.95:
        raise ValueError("the frozen evaluation supports the 95% Wilson interval only")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4 * total**2))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(
    values: list[float], *, seed: int = 3407, resamples: int = 10000
) -> tuple[float, float] | None:
    if not values:
        return None
    clean = [_finite(value, "bootstrap value") for value in values]
    rng = random.Random(seed)
    count = len(clean)
    means = sorted(
        sum(clean[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    low_index = max(0, math.floor(0.025 * resamples))
    high_index = min(resamples - 1, math.ceil(0.975 * resamples) - 1)
    return means[low_index], means[high_index]


def numeric_summary(values: list[Any], *, total: int | None = None) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    expected = len(values) if total is None else total
    if not clean:
        return {
            "n": 0,
            "missing": expected,
            "mean": None,
            "median": None,
            "sample_std": None,
            "min": None,
            "max": None,
            "bootstrap_95_mean": None,
        }
    return {
        "n": len(clean),
        "missing": expected - len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "sample_std": statistics.stdev(clean) if len(clean) >= 2 else None,
        "min": min(clean),
        "max": max(clean),
        "bootstrap_95_mean": bootstrap_mean_interval(clean),
    }


def _matched_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return row["door_pose_id"], int(row["seed"]), row["fixed_or_randomized"]


def paired_comparison(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    left_by_key = {_matched_key(row): row for row in left}
    right_by_key = {_matched_key(row): row for row in right}
    if set(left_by_key) != set(right_by_key) or len(left_by_key) != 36:
        raise UnifiedEvalError(f"{label} does not have 36 matched rollout keys")
    keys = sorted(left_by_key)
    success_diffs = [
        int(bool(right_by_key[key]["success"])) - int(bool(left_by_key[key]["success"]))
        for key in keys
    ]
    metrics = (
        "final_angle_rad",
        "total_ticks",
        "contact_ticks",
        "mean_contact_force_n",
        "max_contact_force_n",
        "p95_contact_force_n",
        "contact_force_impulse_ns",
        "adapter_corrected",
        "adapter_rejected",
        "adapter_warning_count",
    )
    continuous: dict[str, Any] = {}
    for metric in metrics:
        differences = [
            float(right_by_key[key][metric]) - float(left_by_key[key][metric])
            for key in keys
            if right_by_key[key][metric] is not None and left_by_key[key][metric] is not None
        ]
        continuous[metric] = {
            "n_pairs": len(differences),
            "missing_pairs": len(keys) - len(differences),
            "mean_difference_right_minus_left": (
                statistics.fmean(differences) if differences else None
            ),
            "bootstrap_95_mean_difference": bootstrap_mean_interval(differences),
        }
    return {
        "label": label,
        "left_run_id": left[0]["run_id"],
        "right_run_id": right[0]["run_id"],
        "n_pairs": len(keys),
        "success": {
            "mean_difference_right_minus_left": statistics.fmean(success_diffs),
            "right_wins": sum(value == 1 for value in success_diffs),
            "left_wins": sum(value == -1 for value in success_diffs),
            "ties": sum(value == 0 for value in success_diffs),
            "bootstrap_95_mean_difference": bootstrap_mean_interval(success_diffs),
        },
        "continuous": continuous,
    }


def _safety_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warning_counts: dict[str, int] = {}
    review_reasons: list[str] = []
    fail_reasons: list[str] = []
    total_rejected = sum(int(row["adapter_rejected"]) for row in rows)
    total_decisions = sum(int(row["total_ticks"]) for row in rows)
    for row in rows:
        records = json.loads(row["adapter_warning_records"])
        if len(records) > 11:
            review_reasons.append(
                f"seed {row['seed']} has {len(records)} warnings, above v3 envelope"
            )
        for record in records:
            family = str(record["id"])
            warning_counts[family] = warning_counts.get(family, 0) + 1
            if family in {"adapter.invalid_frame", "adapter.non_finite_state"}:
                fail_reasons.append(f"unsafe warning family {family}")
            elif family != "a2.joint_velocity_limit":
                review_reasons.append(f"warning family {family} requires review")
            else:
                evidence = record.get("evidence") or {}
                if (
                    evidence.get("tick_index", 10**9) > 20
                    or evidence.get("exceedance_rad_s", math.inf) > 2.5
                    or evidence.get("consecutive_ticks", 10**9) > 2
                    or evidence.get("duration_ticks", 10**9) > 2
                    or evidence.get("count", 10**9) > 4
                ):
                    review_reasons.append("a2.joint_velocity_limit exceeded v3 envelope")
        if row["force_exceeds_admission_bound"]:
            review_reasons.append(f"seed {row['seed']} exceeded the 200 N watch bound")
        if row["env_truncated"]:
            review_reasons.append(f"seed {row['seed']} was environment-truncated")
    rejection_fraction = total_rejected / total_decisions if total_decisions else 0.0
    if rejection_fraction >= 0.02:
        fail_reasons.append(f"systematic rejection fraction {rejection_fraction:.6f} >= 0.02")
    elif total_rejected:
        review_reasons.append(f"nonzero adapter rejections: {total_rejected}")
    status = "FAIL" if fail_reasons else "REVIEW_REQUIRED" if review_reasons else "PASS"
    return {
        "status": status,
        "warning_family_counts": dict(sorted(warning_counts.items())),
        "total_rejected": total_rejected,
        "rejection_fraction": rejection_fraction,
        "force_bound_exceedance_rollouts": sum(
            bool(row["force_exceeds_admission_bound"]) for row in rows
        ),
        "fail_reasons": sorted(set(fail_reasons)),
        "review_reasons": sorted(set(review_reasons)),
    }


def aggregate_results(plan: UnifiedPlan, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append(row)
    cell_summaries: dict[str, Any] = {}
    metrics = (
        "final_angle_rad",
        "total_ticks",
        "ticks_to_success",
        "contact_ticks",
        "mean_contact_force_n",
        "max_contact_force_n",
        "p95_contact_force_n",
        "contact_force_impulse_ns",
        "adapter_accepted",
        "adapter_corrected",
        "adapter_rejected",
        "adapter_warning_count",
    )
    for run_id, run_rows in sorted(by_run.items()):
        successes = sum(bool(row["success"]) for row in run_rows)
        interval = wilson_interval(successes, len(run_rows))
        cell_summaries[run_id] = {
            "policy": run_rows[0]["policy"],
            "action_space": run_rows[0]["action_space"],
            "dataset_size": run_rows[0]["dataset_size"],
            "success": {
                "count": successes,
                "total": len(run_rows),
                "rate": successes / len(run_rows),
                "wilson_95": interval,
            },
            "metrics": {
                metric: numeric_summary([row[metric] for row in run_rows]) for metric in metrics
            },
            "failure_labels": _counts(row["failure_label"] for row in run_rows),
            "safety": _safety_summary(run_rows),
        }
    comparisons: list[dict[str, Any]] = []
    for policy in ("act", "diffusion"):
        for size in (50, 100, 250, 500):
            left = _find_rows(by_run, policy, "A2_ee_delta", size)
            right = _find_rows(by_run, policy, "A3_obj_rel_ee_delta", size)
            if left is not None and right is not None:
                comparisons.append(
                    paired_comparison(left, right, label=f"A3 minus A2: {policy} N{size}")
                )
    for space in ("A2_ee_delta", "A3_obj_rel_ee_delta"):
        for size in (50, 100, 250, 500):
            left = _find_rows(by_run, "act", space, size)
            right = _find_rows(by_run, "diffusion", space, size)
            if left is not None and right is not None:
                comparisons.append(
                    paired_comparison(
                        left,
                        right,
                        label=f"Diffusion minus ACT: {space} N{size}",
                    )
                )
    return {
        "schema": "alexdoor_xas.phase3_unified_aggregate.v1",
        "protocol_id": plan.protocol_id,
        "plan_sha256": plan.sha256,
        "attempt_id": plan.attempt_id,
        "source_git_commit": plan.source_git_commit,
        "training_seed": 0,
        "primary_rollouts": len(rows),
        "completed_cells": len(by_run),
        "cells": cell_summaries,
        "paired_comparisons": comparisons,
        "diffusion_diagnostics": {
            "status": "INCOMPLETE",
            "finding": (
                "Returned training evidence uses a 10-step DDIM validation metric, and the "
                "primary matrix freezes DDIM-10/Tp16/Ta8. No separate closed-loop sampler or "
                "horizon diagnostic sweep is available or authorized."
            ),
        },
        "claims_boundary": (
            "Training used seed 0 only. Matched rollout seeds support paired evaluation "
            "comparisons but do not establish robustness across training seeds."
        ),
    }


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        label = "success" if value is None else str(value)
        result[label] = result.get(label, 0) + 1
    return dict(sorted(result.items()))


def _find_rows(
    by_run: dict[str, list[dict[str, Any]]], policy: str, space: str, size: int
) -> list[dict[str, Any]] | None:
    matches = [
        rows
        for rows in by_run.values()
        if rows[0]["policy"] == policy
        and rows[0]["action_space"] == space
        and rows[0]["dataset_size"] == size
    ]
    return matches[0] if len(matches) == 1 else None


def _write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise UnifiedEvalError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        content = temporary.read_bytes()
        if path.exists():
            if path.read_bytes() != content:
                raise UnifiedEvalError(f"refusing to overwrite conflicting CSV: {path}")
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report_markdown(plan: UnifiedPlan, audit: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# Unified Phase 3 Scientific Evaluation",
        "",
        "## Confirmed artifact facts",
        "",
        f"- Gilbreth attempt `{plan.attempt_id}` used source commit `{plan.source_git_commit}`.",
        f"- Completed primary cells: {aggregate['completed_cells']}/16.",
        f"- Included primary rollouts: {aggregate['primary_rollouts']}/576.",
        f"- Excluded cells: {audit['excluded_cells']}.",
        "- All included rows use the frozen D0-D4 seed/pose protocol, adapter-v1, "
        "CPU simulation, CUDA inference, a 45 degree threshold, and 600 ticks.",
        "",
        "## Methods",
        "",
        "Each checkpoint was evaluated on 36 matched rollouts: 20 at D0 and four each "
        "at D1-D4. ACT uses horizon 40 without temporal ensembling. Diffusion uses "
        "DDIM-10 with Tp=16 and Ta=8. Success intervals are 95% Wilson intervals; "
        "continuous and paired mean-difference intervals use 10,000 deterministic "
        "bootstrap resamples with seed 3407. No missing value is imputed.",
        "",
        "## Cell results",
        "",
        "| Run | Success | 95% Wilson CI | Mean final angle (deg) | Peak force (N) | Safety |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for run_id, summary in sorted(
        aggregate["cells"].items(),
        key=lambda item: (
            item[1]["dataset_size"],
            item[1]["policy"],
            item[1]["action_space"],
        ),
    ):
        success = summary["success"]
        interval = success["wilson_95"]
        final_deg = math.degrees(summary["metrics"]["final_angle_rad"]["mean"])
        peak = summary["metrics"]["max_contact_force_n"]["max"]
        lines.append(
            f"| `{run_id}` | {success['count']}/{success['total']} "
            f"({success['rate']:.1%}) | [{interval[0]:.1%}, {interval[1]:.1%}] | "
            f"{final_deg:.2f} | {peak:.2f} | {summary['safety']['status']} |"
        )
    lines.extend(
        [
            "",
            "## Matched comparisons",
            "",
            "Positive values mean the named right-hand method/action space was higher.",
            "",
            "| Comparison | Success difference | 95% paired bootstrap CI | "
            "Right wins | Left wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in aggregate["paired_comparisons"]:
        success = comparison["success"]
        interval = success["bootstrap_95_mean_difference"]
        lines.append(
            f"| {comparison['label']} | {success['mean_difference_right_minus_left']:+.3f} | "
            f"[{interval[0]:+.3f}, {interval[1]:+.3f}] | {success['right_wins']} | "
            f"{success['left_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-size trends",
            "",
            "N50/N100/N250/N500 results are reported descriptively per policy/action-space "
            "cell. Monotonic improvement is claimed only if every observed matched step "
            "supports it; otherwise the trend is explicitly non-monotonic or inconclusive.",
            "",
            "## Contact, adapter safety, and failure modes",
            "",
            "The aggregate JSON preserves per-cell contact/force distributions, adapter "
            "accepted/corrected/rejected counts, structured warning families, force-bound "
            "watch items, and failure-label counts. Adapter evidence consists of per-tick "
            "decision counts plus warning and force-peak decision windows; it is not a full "
            "per-tick adapter trace.",
            "",
            "## Diffusion diagnostics",
            "",
            aggregate["diffusion_diagnostics"]["finding"],
            "",
            "## Supported, inconclusive, and unsupported claims",
            "",
            "Supported findings are limited to the complete matched cells and the frozen "
            "door-pose/orientation benchmark. Exact directions and uncertainty are recorded "
            "in `aggregate_summary.json`.",
            "",
            "Diffusion sampler/horizon sensitivity and robustness across training seeds are "
            "inconclusive. Training used seed 0 only; matched evaluation seeds do not prove "
            "training-seed robustness.",
            "",
            "This evidence does not support VLA readiness, A4 learning, hardware readiness, "
            "general geometry/viewpoint/language generalization, sim-to-real transfer, RL, "
            "WAM-lite, or fake-door claims.",
            "",
            "## Phase 4 planning recommendation",
            "",
            "Use only complete, provenance-valid Phase 3 cells to choose candidate action "
            "representations or baseline families. Treat safety review items and diagnostic "
            "gaps as planning inputs, not readiness evidence. Any Phase 4 execution requires "
            "separate authorization.",
            "",
            "## Artifact paths",
            "",
            f"- Artifact completeness: `{plan.curated_root / 'artifact_completeness.csv'}`",
            f"- Normalized rollouts: `{plan.curated_root / 'normalized_rollouts.csv'}`",
            f"- Aggregate summary: `{plan.curated_root / 'aggregate_summary.json'}`",
            f"- Exclusions: `{plan.curated_root / 'exclusions.json'}`",
            f"- Resolved plan: `{plan.curated_root / 'evaluation_plan.resolved.json'}`",
            "",
            "## Remaining caveats",
            "",
            "One simulated door family, state-only policies, CPU simulation, limited force "
            "sensing, no general collision/slip sensing, and no independent training-seed "
            "replication bound every conclusion.",
        ]
    )
    if audit["exclusions"]:
        lines.extend(["", "## Exclusions", ""])
        for exclusion in audit["exclusions"]:
            lines.append(f"- `{exclusion['run_id']}`: {'; '.join(exclusion['reasons'])}")
    return "\n".join(lines) + "\n"


def write_report_artifacts(plan: UnifiedPlan) -> dict[str, Path]:
    audit = audit_evidence(plan)
    included = audit.pop("_rows")
    if not included:
        raise UnifiedEvalError("no complete primary cell is available for scientific reporting")
    rows = normalize_rows(plan, included)
    aggregate = aggregate_results(plan, rows)
    plan.curated_root.mkdir(parents=True, exist_ok=True)
    completeness_path = plan.curated_root / "artifact_completeness.csv"
    normalized_path = plan.curated_root / "normalized_rollouts.csv"
    aggregate_path = plan.curated_root / "aggregate_summary.json"
    exclusions_path = plan.curated_root / "exclusions.json"
    resolved_path = plan.curated_root / "evaluation_plan.resolved.json"
    report_path = plan.curated_root / "report.md"
    _write_csv_exclusive(completeness_path, audit["cells"])
    _write_csv_exclusive(normalized_path, rows)
    _publish_exclusive_json(aggregate_path, aggregate)
    _publish_exclusive_json(
        exclusions_path,
        {
            "schema": "alexdoor_xas.phase3_unified_exclusions.v1",
            "protocol_id": plan.protocol_id,
            "excluded_cells": audit["excluded_cells"],
            "exclusions": audit["exclusions"],
        },
    )
    resolved_source = plan.workspace_root / "provenance" / "evaluation_plan.resolved.json"
    resolved = json.loads(resolved_source.read_text())
    _publish_exclusive_json(resolved_path, resolved)
    report = _report_markdown(plan, audit, aggregate)
    if report_path.exists():
        if report_path.read_text() != report:
            raise UnifiedEvalError(f"refusing to overwrite conflicting report: {report_path}")
    else:
        report_path.write_text(report)
    return {
        "artifact_completeness": completeness_path,
        "normalized_rollouts": normalized_path,
        "aggregate_summary": aggregate_path,
        "exclusions": exclusions_path,
        "resolved_plan": resolved_path,
        "report": report_path,
    }


__all__ = [
    "CellPlan",
    "PosePlan",
    "UnifiedEvalError",
    "UnifiedPlan",
    "aggregate_results",
    "audit_evidence",
    "bootstrap_mean_interval",
    "build_eval_command",
    "canonical_pose_path",
    "load_plan",
    "normalize_rows",
    "numeric_summary",
    "paired_comparison",
    "prepare_workspace",
    "run_cell",
    "run_preflight",
    "run_pose",
    "sha256_file",
    "validate_eval_payload",
    "verify_immutable_inventory",
    "wilson_interval",
    "write_report_artifacts",
]
