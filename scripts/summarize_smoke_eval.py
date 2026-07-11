#!/usr/bin/env python
"""Merge per-pose smoke-eval JSONs into one summary + metadata-coverage gate (no Kit).

Each smoke checkpoint is evaluated once per door pose (one Isaac process per
pose writes ``metrics/<policy>_eval_<pose>.json`` next to the checkpoint).
This script gathers every eval JSON of the given run directories, verifies the
per-rollout metadata coverage the later unified report depends on (a missing
field is a hard failure — the task's stop condition), and writes a combined
summary with per-pose and overall aggregates per run.

Run through the official launcher (pure Python, no Kit)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/summarize_smoke_eval.py --out outputs/local_smoke_n50/summary.json \
        outputs/door_push_alex_v2/act_door_push/local_smoke_act_a2_n50_seed0 [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_ROW_FIELDS = (
    "seed",
    "randomized",
    "door_pose_id",
    "door_yaw_deg",
    "door_offset_xy",
    "success",
    "failure_label",
    "initial_angle_rad",
    "final_angle_rad",
    "n_ticks",
    "contact_ticks",
    "contact_source",
    "force_n",
    "impulse_ns",
    "contact_unavailable_reason",
    "force_exceeds_admission_bound",
    "n_accepted",
    "n_corrected",
    "n_rejected",
    "n_warnings",
    "warning_counts",
    "notes",
)
REQUIRED_TOP_FIELDS = (
    "checkpoint",
    "action_space",
    "obs_preset",
    "max_ticks",
    "success_angle_deg",
    "base_seed",
    "door_pose",
    "control_dt",
    "dataset_provenance",
    "seed_protocol",
    "rollouts",
    "aggregate",
)
DIFFUSION_TOP_FIELDS = ("horizon", "n_action_steps", "sampler", "num_inference_steps")
DIFFUSION_ROW_FIELDS = ("sampler", "num_inference_steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Smoke run dirs (one per cell).")
    parser.add_argument("--out", type=Path, required=True, help="Summary JSON output path.")
    parser.add_argument(
        "--expected-poses",
        type=str,
        default="D0,D1,D2,D3,D4",
        help="Comma-separated pose ids every run must cover (matrix completeness gate).",
    )
    return parser.parse_args()


def check_coverage(payload: dict, path: Path, problems: list[str]) -> None:
    is_diffusion = "diffusion" in path.name
    top_fields = REQUIRED_TOP_FIELDS + (DIFFUSION_TOP_FIELDS if is_diffusion else ("chunk_size",))
    for field in top_fields:
        if field not in payload:
            problems.append(f"{path}: missing top-level field {field!r}")
    row_fields = REQUIRED_ROW_FIELDS + (DIFFUSION_ROW_FIELDS if is_diffusion else ())
    for i, row in enumerate(payload.get("rollouts", [])):
        missing = [field for field in row_fields if field not in row]
        if missing:
            problems.append(f"{path}: rollout {i} (seed {row.get('seed')}) missing {missing}")
    provenance = payload.get("dataset_provenance") or {}
    if not provenance.get("fingerprint_sha256"):
        problems.append(f"{path}: dataset_provenance has no fingerprint_sha256")
    if payload.get("aggregate", {}).get("fixed_determinism_spread_rad") is None and (
        payload.get("seed_protocol", {}).get("episodes_fixed", 0) > 0
    ):
        problems.append(f"{path}: fixed block present but no determinism spread recorded")


def main() -> int:
    args = parse_args()
    problems: list[str] = []
    runs: dict[str, dict] = {}
    for run_dir in args.run_dirs:
        # Pose-qualified files only: a stale legacy <policy>_eval.json (no pose
        # id) must never be ingested as a phantom default-pose block.
        eval_files = sorted((run_dir / "metrics").glob("*_eval_*.json"))
        if not eval_files:
            problems.append(f"{run_dir}: no eval JSONs under metrics/")
            continue
        poses: dict[str, dict] = {}
        diagnostics: dict[str, dict] = {}
        rows_all: list[dict] = []
        for path in eval_files:
            payload = json.loads(path.read_text())
            check_coverage(payload, path, problems)
            pose_id = (payload.get("door_pose") or {}).get("door_pose_id", "D0")
            if pose_id in poses or pose_id in diagnostics:
                problems.append(f"{run_dir}: duplicate eval files for pose {pose_id!r}")
                continue
            entry = {
                "eval_json": str(path),
                "door_pose": payload.get("door_pose"),
                "base_seed": payload.get("base_seed"),
                "sampler": payload.get("sampler"),
                "num_inference_steps": payload.get("num_inference_steps"),
                "aggregate": payload.get("aggregate"),
            }
            if "diag" in pose_id:
                # Diagnostics (e.g. the DDPM-100 hard-seed probe) are reported
                # separately and never enter the main matrix aggregates.
                diagnostics[pose_id] = entry
                continue
            poses[pose_id] = entry
            rows_all.extend(payload.get("rollouts", []))
        expected_poses = {pose.strip() for pose in args.expected_poses.split(",") if pose.strip()}
        if set(poses) != expected_poses:
            problems.append(
                f"{run_dir}: pose matrix incomplete — got {sorted(poses)}, "
                f"expected {sorted(expected_poses)}"
            )
        n_success = sum(1 for row in rows_all if row.get("success"))
        # Run-level policy metadata must come from a primary (non-diag) file.
        primary_files = [
            path
            for path in eval_files
            if "diag" not in (
                json.loads(path.read_text()).get("door_pose") or {}
            ).get("door_pose_id", "D0")
        ]
        sample = json.loads((primary_files or eval_files)[0].read_text())
        runs[run_dir.name] = {
            "run_dir": str(run_dir),
            "checkpoint": sample.get("checkpoint"),
            "action_space": sample.get("action_space"),
            "obs_preset": sample.get("obs_preset"),
            "policy_metadata": {
                key: sample.get(key)
                for key in (
                    "chunk_size",
                    "temporal_ensemble",
                    "horizon",
                    "n_action_steps",
                    "sampler",
                    "num_inference_steps",
                )
                if key in sample
            },
            "dataset_fingerprint_sha256": (sample.get("dataset_provenance") or {}).get(
                "fingerprint_sha256"
            ),
            "poses": dict(sorted(poses.items())),
            "diagnostics": dict(sorted(diagnostics.items())),
            "overall": {
                "n_rollouts": len(rows_all),
                "n_success": n_success,
                "success_rate": (n_success / len(rows_all)) if rows_all else None,
                "adapter": {
                    "n_accepted": sum(int(row.get("n_accepted", 0)) for row in rows_all),
                    "n_corrected": sum(int(row.get("n_corrected", 0)) for row in rows_all),
                    "n_rejected": sum(int(row.get("n_rejected", 0)) for row in rows_all),
                    "n_warnings": sum(int(row.get("n_warnings", 0)) for row in rows_all),
                },
                "failure_labels": sorted(
                    {row["failure_label"] for row in rows_all if row.get("failure_label")}
                ),
                "n_force_exceeds_admission_bound": sum(
                    1 for row in rows_all if row.get("force_exceeds_admission_bound")
                ),
            },
        }

    summary = {
        "schema": "alexdoor_xas.smoke_eval_summary.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "metadata_coverage": "PASS" if not problems else "FAIL",
        "problems": problems,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    for name, run in runs.items():
        overall = run["overall"]
        print(
            f"[{name}] rollouts={overall['n_rollouts']} "
            f"success={overall['n_success']}/{overall['n_rollouts']} "
            f"rejected={overall['adapter']['n_rejected']} "
            f"warnings={overall['adapter']['n_warnings']} poses={list(run['poses'])}",
            flush=True,
        )
    if problems:
        for problem in problems:
            print(f"[fail] {problem}", flush=True)
        print(f"FAIL: metadata coverage incomplete ({args.out})", flush=True)
        return 1
    print(f"PASS: smoke eval summary written ({args.out})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
