"""Closed-loop metrics, traces, and immutable evaluation runs."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.assets.door_scene import CANONICAL_DOOR_POSES
from alexdoor_xas.policies.common.runs import (
    RUN_FORMAT,
    load_resolved_config,
    write_json_atomic,
    write_run_report,
)


def validate_evaluation_protocol(protocol: dict[str, Any], policy: str) -> None:
    """Reject incomplete or internally inconsistent evaluation protocols."""
    required = {
        "poses",
        "rollout_count",
        "success_threshold_deg",
        "force_limit_n",
        "horizon_ticks",
        "control",
        "policy_execution",
    }
    if set(protocol) != required:
        raise ValueError(
            "evaluation protocol fields must be exactly " + ", ".join(sorted(required))
        )
    if policy not in {"act", "diffusion"}:
        raise ValueError(f"unknown policy {policy!r}")
    if not protocol["poses"]:
        raise ValueError("evaluation protocol requires at least one pose")
    pose_ids: set[str] = set()
    for pose in protocol["poses"]:
        if set(pose) != {
            "pose",
            "yaw_rad",
            "xy_offset_m",
            "fixed_seeds",
            "randomized_seeds",
        }:
            raise ValueError("each protocol pose must contain only canonical pose fields")
        pose_id = str(pose["pose"])
        canonical = CANONICAL_DOOR_POSES.get(pose_id)
        if canonical is None:
            raise ValueError(f"unknown canonical pose {pose_id!r}")
        if float(pose["yaw_rad"]) != canonical.yaw_rad or [
            float(value) for value in pose["xy_offset_m"]
        ] != list(canonical.xy_offset_m):
            raise ValueError(f"protocol transform does not match canonical pose {pose_id}")
        if pose_id in pose_ids:
            raise ValueError(f"duplicate protocol pose {pose_id!r}")
        pose_ids.add(pose_id)
        fixed = [int(seed) for seed in pose["fixed_seeds"]]
        randomized = [int(seed) for seed in pose["randomized_seeds"]]
        if len(fixed + randomized) != len(set(fixed + randomized)):
            raise ValueError(f"protocol pose {pose_id} contains duplicate seeds")
        if any(seed < 0 for seed in fixed + randomized):
            raise ValueError("protocol seeds must be non-negative")
    protocol_rollouts(protocol)
    for field in ("success_threshold_deg", "force_limit_n", "horizon_ticks"):
        if float(protocol[field]) <= 0:
            raise ValueError(f"evaluation protocol {field} must be positive")
    execution_fields = (
        {"temporal_ensemble", "ensemble_m"}
        if policy == "act"
        else {"n_action_steps", "sampler", "num_inference_steps"}
    )
    if set(protocol["policy_execution"]) != execution_fields:
        raise ValueError(
            f"{policy} policy_execution fields must be exactly "
            + ", ".join(sorted(execution_fields))
        )
    control_fields = {
        "sim_dt_s",
        "decimation",
        "max_position_delta_m",
        "max_rotation_delta_rad",
        "adapter",
        "contact_entry_shaping",
        "stop_on_reject",
    }
    if set(protocol["control"]) != control_fields:
        raise ValueError(
            "protocol control fields must be exactly " + ", ".join(sorted(control_fields))
        )
    if protocol["control"]["adapter"] != "adapter-v1":
        raise ValueError("only adapter-v1 is supported")
    for field in ("sim_dt_s", "decimation", "max_position_delta_m", "max_rotation_delta_rad"):
        if float(protocol["control"][field]) <= 0:
            raise ValueError(f"protocol control.{field} must be positive")
    execution = protocol["policy_execution"]
    if policy == "act" and float(execution["ensemble_m"]) <= 0:
        raise ValueError("ACT ensemble_m must be positive")
    if policy == "diffusion":
        if int(execution["n_action_steps"]) <= 0 or int(execution["num_inference_steps"]) <= 0:
            raise ValueError("Diffusion action and inference step counts must be positive")
        if execution["sampler"] not in {"ddpm", "ddim"}:
            raise ValueError("Diffusion sampler must be ddpm or ddim")


def protocol_rollouts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a frozen protocol into deterministic pose/seed/subset keys."""
    items: list[dict[str, Any]] = []
    for pose in protocol["poses"]:
        for seed in pose["fixed_seeds"]:
            items.append({"pose": pose["pose"], "seed": int(seed), "status": "fixed"})
        for seed in pose["randomized_seeds"]:
            items.append({"pose": pose["pose"], "seed": int(seed), "status": "randomized"})
    if len(items) != int(protocol["rollout_count"]):
        raise ValueError("evaluation protocol rollout_count does not match its seed lists")
    return items


def rollout_key(pose: str, seed: int, status: str) -> str:
    return f"{pose}_seed{int(seed)}_{status}"


def warning_family_counts(decisions) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        for warning in getattr(decision, "warning_records", ()) or ():
            counts[str(warning.id)] += 1
    return dict(sorted(counts.items()))


def _termination_reason(reason: str) -> str:
    mapped = {
        "success": "controller_done",
        "policy_exhausted": "controller_done",
        "rejection_stop": "step_error",
        "invalid_simulator_state": "step_error",
    }.get(reason, reason)
    allowed = {
        "controller_done",
        "controller_timeout",
        "tick_budget",
        "environment_terminated",
        "environment_truncated",
        "step_error",
    }
    return mapped if mapped in allowed else "step_error"


def factual_rollout_row(
    *,
    pose: str,
    seed: int,
    status: str,
    result,
    control_dt_s: float,
    force_limit_n: float,
) -> tuple[dict[str, Any], list[float]]:
    """Build one factual row plus its private force samples for pooled aggregation."""
    forces = [float(value) for value in result.force_n_per_tick if value is not None]
    if forces and not np.isfinite(forces).all():
        raise RuntimeError("non-finite rollout force samples cannot become metrics")
    accepted = int(result.log.n_accepted)
    corrected = int(result.log.n_corrected)
    rejected = int(result.log.n_rejected)
    decision_count = accepted + corrected + rejected
    exceedance_count = sum(value > force_limit_n for value in forces)
    row = {
        "rollout_key": rollout_key(pose, seed, status),
        "pose": pose,
        "seed": int(seed),
        "status": status,
        "success": bool(result.success),
        "termination_reason": _termination_reason(str(result.termination_reason)),
        "environment_terminated": bool(result.environment_terminated),
        "environment_truncated": bool(result.environment_truncated),
        "evaluated_steps": int(result.n_ticks),
        "time_to_success_s": (
            None
            if result.first_success_tick is None
            else float(result.first_success_tick * control_dt_s)
        ),
        "contact_force_n": {
            "mean": float(np.mean(forces)) if forces else None,
            "p95": float(np.percentile(forces, 95)) if forces else None,
            "maximum": max(forces) if forces else None,
        },
        "impulse_ns": float(sum(forces) * control_dt_s) if forces else None,
        "force_limit_n": float(force_limit_n),
        "force_limit_exceedance_count": exceedance_count,
        "adapter": {
            "decision_count": decision_count,
            "accepted_count": accepted,
            "corrected_count": corrected,
            "rejected_count": rejected,
            "accepted_rate": accepted / decision_count if decision_count else None,
            "corrected_rate": corrected / decision_count if decision_count else None,
            "rejected_rate": rejected / decision_count if decision_count else None,
        },
        "warning_family_counts": warning_family_counts(result.decisions_per_tick),
    }
    return row, forces


def aggregate_closed_loop(
    rows: list[dict[str, Any]], force_samples: dict[str, list[float]]
) -> dict[str, Any]:
    """Aggregate overall, pose, subset, and pose-plus-subset factual metrics."""
    if not rows:
        raise ValueError("closed-loop aggregation requires at least one rollout")

    def selected(predicate) -> list[dict[str, Any]]:
        return [row for row in rows if predicate(row)]

    by_pose = {
        pose: _aggregate_subset(
            selected(lambda row, value=pose: row["pose"] == value), force_samples
        )
        for pose in sorted({row["pose"] for row in rows})
    }
    by_status = {
        status: _aggregate_subset(
            selected(lambda row, value=status: row["status"] == value), force_samples
        )
        for status in ("fixed", "randomized")
        if any(row["status"] == status for row in rows)
    }
    by_pose_status = {
        pose: {
            status: _aggregate_subset(
                selected(
                    lambda row, pose_value=pose, status_value=status: (
                        row["pose"] == pose_value and row["status"] == status_value
                    )
                ),
                force_samples,
            )
            for status in ("fixed", "randomized")
            if any(row["pose"] == pose and row["status"] == status for row in rows)
        }
        for pose in by_pose
    }
    return {
        "overall": _aggregate_subset(rows, force_samples),
        "by_pose": by_pose,
        "by_status": by_status,
        "by_pose_and_status": by_pose_status,
    }


def _aggregate_subset(
    rows: list[dict[str, Any]], force_samples: dict[str, list[float]]
) -> dict[str, Any]:
    success_rows = [row for row in rows if row["success"]]
    time_to_success = [
        float(row["time_to_success_s"])
        for row in success_rows
        if row["time_to_success_s"] is not None
    ]
    forces = [value for row in rows for value in force_samples.get(row["rollout_key"], [])]
    decisions = sum(int(row["adapter"]["decision_count"]) for row in rows)
    adapter_counts = {
        name: sum(int(row["adapter"][f"{name}_count"]) for row in rows)
        for name in ("accepted", "corrected", "rejected")
    }
    warning_counts: Counter[str] = Counter()
    for row in rows:
        warning_counts.update(row["warning_family_counts"])
    return {
        "rollout_count": len(rows),
        "success_count": len(success_rows),
        "success_rate": len(success_rows) / len(rows),
        "time_to_success_s": {
            "sample_count": len(time_to_success),
            "median": float(np.median(time_to_success)) if time_to_success else None,
            "p90": float(np.percentile(time_to_success, 90)) if time_to_success else None,
        },
        "contact_force_n": {
            "sample_count": len(forces),
            "mean": float(np.mean(forces)) if forces else None,
            "p95": float(np.percentile(forces, 95)) if forces else None,
            "maximum": max(forces) if forces else None,
        },
        "impulse_ns": sum(
            float(row["impulse_ns"]) for row in rows if row["impulse_ns"] is not None
        ),
        "force_limit_exceedance_count": sum(
            int(row["force_limit_exceedance_count"]) for row in rows
        ),
        "adapter": {
            "decision_count": decisions,
            **{f"{name}_count": count for name, count in adapter_counts.items()},
            **{
                f"{name}_rate": count / decisions if decisions else None
                for name, count in adapter_counts.items()
            },
        },
        "warning_family_counts": dict(sorted(warning_counts.items())),
    }


def write_closed_loop_summary(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the required three-panel factual summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["rollout_key"] for row in rows]
    x = np.arange(len(rows))
    times = [
        np.nan if row["time_to_success_s"] is None else row["time_to_success_s"] for row in rows
    ]
    peaks = [
        np.nan if row["contact_force_n"]["maximum"] is None else row["contact_force_n"]["maximum"]
        for row in rows
    ]
    corrected = [row["adapter"]["corrected_rate"] or 0.0 for row in rows]
    rejected = [row["adapter"]["rejected_rate"] or 0.0 for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(x, times)
    axes[0].set_ylabel("time to success (s)")
    axes[1].bar(x, peaks)
    axes[1].axhline(rows[0]["force_limit_n"], color="red", linestyle="--", label="force limit")
    axes[1].set_ylabel("peak force (N)")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, corrected, label="corrected")
    axes[2].bar(x, rejected, bottom=corrected, label="rejected")
    axes[2].set_ylabel("adapter rate")
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(x, labels, rotation=90, fontsize=6)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def trace_required(row: dict[str, Any], selected_keys: set[str] | None = None) -> bool:
    return (
        not row["success"]
        or row["force_limit_exceedance_count"] > 0
        or row["rollout_key"] in (selected_keys or set())
    )


def closed_loop_trace_payload(result) -> dict[str, Any]:
    """Detailed factual trace retained only when the selective rule asks for it."""
    return {
        "evaluated_steps": int(result.n_ticks),
        "termination_reason": _termination_reason(str(result.termination_reason)),
        "environment_terminated": bool(result.environment_terminated),
        "environment_truncated": bool(result.environment_truncated),
        "requested_actions": [
            None
            if decision.requested is None
            else np.asarray(decision.requested, dtype=np.float64).tolist()
            for decision in result.decisions_per_tick
        ],
        "applied_actions": [
            None
            if decision.applied is None
            else np.asarray(decision.applied, dtype=np.float64).tolist()
            for decision in result.decisions_per_tick
        ],
        "adapter_status": [str(decision.status) for decision in result.decisions_per_tick],
        "contact": [None if value is None else bool(value) for value in result.contact_per_tick],
        "force_n": [None if value is None else float(value) for value in result.force_n_per_tick],
    }


def write_selected_traces(
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    trace_payloads: dict[str, dict[str, Any]],
    selected_keys: set[str] | None = None,
) -> list[str]:
    """Create traces/ only when at least one rollout meets the retention rule."""
    row_keys = {row["rollout_key"] for row in rows}
    unknown = sorted((selected_keys or set()) - row_keys)
    if unknown:
        raise ValueError(f"explicit trace rollout keys were not evaluated: {unknown}")
    selected = [row for row in rows if trace_required(row, selected_keys)]
    if not selected:
        return []
    trace_dir = Path(run_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=False)
    retained: list[str] = []
    for row in selected:
        key = row["rollout_key"]
        payload = trace_payloads.get(key)
        if payload is None:
            raise ValueError(f"selected rollout {key!r} has no trace payload")
        path = write_json_atomic(trace_dir / f"{key}.json", payload)
        retained.append(str(path.relative_to(run_dir)))
    return retained


def evaluation_preflight(
    *,
    source_checkpoint: str | Path,
    requested_protocol: dict[str, Any],
    policy: str,
) -> tuple[Path, dict[str, Any]]:
    """Validate the checkpoint, training config, and requested protocol."""
    checkpoint = Path(source_checkpoint).expanduser().resolve()
    source_run = checkpoint.parent.parent
    if checkpoint.name != "best.pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError("source checkpoint must be <training-run>/checkpoints/best.pt")
    if not checkpoint.is_file():
        raise ValueError(f"source checkpoint does not exist: {checkpoint}")
    source_resolved = load_resolved_config(source_run)
    if source_resolved.get("run_type") != "training" or source_resolved.get("policy") != policy:
        raise ValueError("source checkpoint does not belong to the expected training run")
    try:
        if policy == "act":
            from alexdoor_xas.policies.act.config import act_config_from_dict

            act_config_from_dict(source_resolved.get("config"))
        elif policy == "diffusion":
            from alexdoor_xas.policies.diffusion.config import diffusion_config_from_dict

            diffusion_config_from_dict(source_resolved.get("config"))
        else:
            raise ValueError(f"unknown policy {policy!r}")
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid training config: {error}") from error
    validate_evaluation_protocol(requested_protocol, policy)
    return source_run, source_resolved


def _allocate_evaluation_directory(source_run: Path) -> tuple[str, Path]:
    parent = source_run / "closed_loop"
    parent.mkdir(exist_ok=True)
    base = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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


def prepare_evaluation_run(
    *,
    source_checkpoint: str | Path,
    requested_protocol: dict[str, Any],
    policy: str,
) -> tuple[Path, dict[str, Any]]:
    """Allocate one immutable evaluation below its training run."""
    source_run, source_resolved = evaluation_preflight(
        source_checkpoint=source_checkpoint,
        requested_protocol=requested_protocol,
        policy=policy,
    )
    checkpoint = Path(source_checkpoint).expanduser().resolve()
    run_id, run_dir = _allocate_evaluation_directory(source_run)
    resolved = {
        "format": RUN_FORMAT,
        "run_type": "evaluation",
        "run_id": run_id,
        "policy": policy,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_run_id": source_resolved["run_id"],
        "checkpoint": str(checkpoint),
        "config": source_resolved["config"],
        "evaluation_protocol": requested_protocol,
    }
    write_json_atomic(run_dir / "resolved_config.json", resolved, exclusive=True)
    return run_dir, resolved


def publish_closed_loop(
    *,
    run_dir: str | Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
    force_samples: dict[str, list[float]],
    trace_payloads: dict[str, dict[str, Any]] | None = None,
    selected_trace_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Publish one immutable evaluation result."""
    run_dir = Path(run_dir)
    selected_trace_keys = selected_trace_keys or set()
    row_keys = {row["rollout_key"] for row in rows}
    unknown = sorted(selected_trace_keys - row_keys)
    if unknown:
        raise ValueError(f"explicit trace rollout keys were not evaluated: {unknown}")
    selected_rows = [row for row in rows if trace_required(row, selected_trace_keys)]
    trace_payloads = trace_payloads or {}
    missing_traces = [
        row["rollout_key"] for row in selected_rows if row["rollout_key"] not in trace_payloads
    ]
    if missing_traces:
        raise ValueError(f"selected rollouts have no trace payload: {missing_traces}")
    outputs = [run_dir / name for name in ("metrics.json", "summary.png", "report.md", "traces")]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")
    aggregate = aggregate_closed_loop(rows, force_samples)
    metrics = {
        "protocol": resolved["evaluation_protocol"],
        "rollouts": rows,
        "aggregate": aggregate,
    }
    write_json_atomic(run_dir / "metrics.json", metrics, exclusive=True)
    write_closed_loop_summary(rows, run_dir / "summary.png")

    retained = write_selected_traces(
        run_dir,
        rows,
        trace_payloads,
        selected_trace_keys,
    )
    write_run_report(
        run_dir,
        resolved,
        status="completed",
        closed_loop=metrics,
        source_checkpoint=resolved["checkpoint"],
        retained_optional_artifacts=retained,
    )
    return metrics
