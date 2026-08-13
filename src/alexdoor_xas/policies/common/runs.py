"""Canonical learned-policy run allocation, state, plotting, and reports."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.assets.door_scene import CANONICAL_DOOR_POSES

RUN_FORMAT = "alexdoor_xas.run.v1"
POLICY_DIRS = ("act", "diffusion")
FORCE_LIMIT_N = 200.0
POSE_PROTOCOL = (
    ("D0", 100, 5, 15),
    ("D1", 200, 1, 3),
    ("D2", 210, 1, 3),
    ("D3", 220, 1, 3),
    ("D4", 230, 1, 3),
)


def action_token(action_space: str) -> str:
    if action_space == A2_EE_DELTA:
        return "a2"
    if action_space == A3_OBJ_REL_EE_DELTA:
        return "a3"
    raise ValueError(f"learned runs require A2 or A3, got {action_space!r}")


def dataset_token(version: str, view_id: str | None) -> str:
    """Compact a dataset/view identifier for the filesystem-safe run ID."""
    source = view_id or version
    match = re.fullmatch(r"v(\d+)(?:_[a-zA-Z0-9]+)*_n(\d+)", source)
    token = f"v{match.group(1)}n{match.group(2)}" if match else source
    token = re.sub(r"[^a-zA-Z0-9]+", "", token)
    if not token:
        raise ValueError(f"dataset identifier cannot produce a run token: {source!r}")
    return token.lower()


def learned_run_parent(output_root: str | Path | None, policy: str) -> Path:
    if policy not in POLICY_DIRS:
        raise ValueError(f"unknown learned policy {policy!r}")
    root = Path(output_root).expanduser().resolve() if output_root else paths.OUTPUTS_DIR
    return root / paths.ALEX_V2_TASK / policy


def allocate_run_directory(
    *,
    output_root: str | Path | None,
    policy: str,
    action_space: str,
    dataset_version: str,
    dataset_view_id: str | None,
    seed: int,
    now: datetime | None = None,
) -> tuple[str, Path]:
    """Exclusively allocate a full-timestamp run directory without overwriting."""
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = (
        f"{timestamp}_{action_token(action_space)}_"
        f"{dataset_token(dataset_version, dataset_view_id)}_seed{int(seed)}"
    )
    parent = learned_run_parent(output_root, policy)
    parent.mkdir(parents=True, exist_ok=True)
    revision = 1
    while True:
        run_id = base if revision == 1 else f"{base}_r{revision}"
        run_dir = parent / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            revision += 1
            continue
        return run_id, run_dir


def resolve_resume_directory(path: str | Path, policy: str) -> Path:
    """Validate an explicit resumable run directory."""
    run_dir = Path(path).expanduser().resolve()
    if run_dir.parent.name != policy or run_dir.parent.parent.name != paths.ALEX_V2_TASK:
        raise ValueError(f"resume path must be under door_push_alex_v2/{policy}/: {run_dir}")
    required = (run_dir / "resolved_config.json", run_dir / "checkpoints" / "last.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "resume requires an incomplete run with resolved_config.json and last.pt; "
            f"missing {missing}"
        )
    return run_dir


def frozen_evaluation_protocol(policy: str, rollout_cfg: Any) -> dict[str, Any]:
    """Return the complete default 36-rollout protocol frozen into a training run."""
    poses: list[dict[str, Any]] = []
    for pose_id, base_seed, n_fixed, n_randomized in POSE_PROTOCOL:
        pose = CANONICAL_DOOR_POSES[pose_id]
        poses.append(
            {
                "pose": pose_id,
                "yaw_rad": pose.yaw_rad,
                "xy_offset_m": list(pose.xy_offset_m),
                "fixed_seeds": list(range(base_seed, base_seed + n_fixed)),
                "randomized_seeds": list(
                    range(base_seed + n_fixed, base_seed + n_fixed + n_randomized)
                ),
            }
        )
    execution = (
        {
            "temporal_ensemble": bool(rollout_cfg.temporal_ensemble),
            "ensemble_m": float(rollout_cfg.ensemble_m),
        }
        if policy == "act"
        else {
            "n_action_steps": int(rollout_cfg.n_action_steps),
            "sampler": str(rollout_cfg.sampler),
            "num_inference_steps": int(rollout_cfg.num_inference_steps),
        }
    )
    return {
        "poses": poses,
        "rollout_count": sum(
            len(pose["fixed_seeds"]) + len(pose["randomized_seeds"]) for pose in poses
        ),
        "success_threshold_deg": float(rollout_cfg.success_angle_deg),
        "force_limit_n": FORCE_LIMIT_N,
        "horizon_ticks": int(rollout_cfg.max_ticks),
        "control": {
            "sim_dt_s": 1.0 / 120.0,
            "decimation": 2,
            "max_position_delta_m": 0.02,
            "max_rotation_delta_rad": 0.05,
            "adapter": "adapter-v1",
            "contact_entry_shaping": True,
            "stop_on_reject": False,
        },
        "policy_execution": execution,
    }


def resolved_training_config(
    *,
    run_id: str,
    policy: str,
    config: Any,
    created_utc: str | None = None,
) -> dict[str, Any]:
    config_payload = asdict(config) if is_dataclass(config) else dict(config)
    return {
        "format": RUN_FORMAT,
        "run_type": "training",
        "run_id": run_id,
        "policy": policy,
        "created_utc": created_utc or datetime.now(UTC).isoformat(),
        "config": _jsonable(config_payload),
        "evaluation_protocol": frozen_evaluation_protocol(policy, config.rollout),
    }


def write_json_atomic(path: str | Path, payload: Any, *, exclusive: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and target.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_resolved_config(run_dir: str | Path) -> dict[str, Any]:
    payload = json.loads((Path(run_dir) / "resolved_config.json").read_text())
    if payload.get("format") != RUN_FORMAT:
        raise ValueError(f"unsupported run format: {payload.get('format')!r}")
    return payload


def torch_save_atomic(path: str | Path, payload: Any) -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def capture_rng_states() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def write_training_summary(policy: str, history: dict[str, Any], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epochs"]
    x = [entry["epoch"] + 1 for entry in epochs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    if policy == "act":
        axes[0].plot(x, [entry["train_l1"] for entry in epochs], label="train L1")
        _plot_optional(axes[0], x, epochs, "validation_l1", "validation L1")
        axes[0].set_ylabel("L1")
        axes[1].plot(x, [entry["kl"] for entry in epochs])
        axes[1].set_ylabel("KL")
        axes[2].plot(x, [entry["total_loss"] for entry in epochs])
        axes[2].set_ylabel("total loss")
    else:
        axes[0].plot(x, [entry["train_mse"] for entry in epochs])
        axes[0].set_ylabel("train MSE")
        _plot_optional(axes[1], x, epochs, "sampled_validation_l1", "sampled validation L1")
        axes[1].set_ylabel("validation L1")
        axes[2].plot(x, [entry["learning_rate"] for entry in epochs])
        axes[2].set_ylabel("learning rate")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def write_run_report(
    run_dir: str | Path,
    resolved: dict[str, Any],
    *,
    status: str,
    training: dict[str, Any] | None = None,
    open_loop: dict[str, Any] | None = None,
    closed_loop: dict[str, Any] | None = None,
    anomalies: list[str] | None = None,
    source_checkpoint: str | None = None,
    retained_optional_artifacts: list[str] | None = None,
) -> Path:
    """Write the only narrative artifact for training and evaluation runs."""
    run_dir = Path(run_dir)
    config = resolved.get("config", {})
    dataset = config.get("dataset", {})
    lines = [
        "# Run report",
        "",
        (
            f"- Run: `{resolved['run_id']}`; type: `{resolved['run_type']}`; "
            f"policy: `{resolved['policy']}`; status: `{status}`."
        ),
        (
            f"- Dataset/action: `{dataset.get('task', 'not applicable')}` / "
            f"`{dataset.get('space', 'not applicable')}` / "
            f"`{dataset.get('view_id') or dataset.get('version', 'not applicable')}`."
        ),
        (
            f"- Frozen evaluation protocol: "
            f"{resolved['evaluation_protocol']['rollout_count']} rollouts, success threshold "
            f"{resolved['evaluation_protocol']['success_threshold_deg']:g} deg, force limit "
            f"{resolved['evaluation_protocol']['force_limit_n']:g} N, horizon "
            f"{resolved['evaluation_protocol']['horizon_ticks']} ticks."
        ),
    ]
    if source_checkpoint:
        lines.append(f"- Source checkpoint: `{source_checkpoint}`.")
    if training:
        lines.append(
            f"- Training: best epoch {training.get('best_epoch')}; best validation value "
            f"{_fmt(training.get('best_value'))}; duration "
            f"{_fmt(training.get('duration_s'))} s."
        )
    if open_loop:
        lines.append(
            f"- Open loop: translation L1 mean "
            f"{_fmt(open_loop.get('aggregate_l1_mean'))} over "
            f"{open_loop.get('evaluated_steps')} evaluated steps; representative episode "
            f"`{open_loop.get('representative_episode_id')}`."
        )
    if closed_loop:
        overall = closed_loop.get("aggregate", {}).get("overall", {})
        lines.append(
            f"- Closed loop: {overall.get('success_count', 0)}/"
            f"{overall.get('rollout_count', 0)} successful "
            f"({_fmt(overall.get('success_rate'))})."
        )
    anomaly_values = list(anomalies or [])
    if (
        run_dir / "error.log"
    ).is_file() and "Historical error.log retained after resume." not in anomaly_values:
        anomaly_values.append("Historical error.log retained after resume.")
    lines.append("- Anomalies: " + ("; ".join(anomaly_values) if anomaly_values else "none."))
    optional = retained_optional_artifacts or []
    lines.append(
        "- Retained optional artifacts: "
        + (", ".join(f"`{item}`" for item in optional) if optional else "none.")
    )
    target = run_dir / "report.md"
    target.write_text("\n".join(lines) + "\n")
    return target


def _plot_optional(axis, x: list[int], epochs: list[dict[str, Any]], key: str, label: str) -> None:
    points = [
        (epoch, entry[key])
        for epoch, entry in zip(x, epochs, strict=True)
        if entry[key] is not None
    ]
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points], label=label)


def _fmt(value: Any) -> str:
    return "not available" if value is None else f"{float(value):.5g}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
