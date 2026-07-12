#!/usr/bin/env python
"""Merge per-pose smoke-eval JSONs into one summary with three separate gates (no Kit).

Each smoke checkpoint is evaluated once per door pose (one Isaac process per
pose writes ``metrics/<policy>_eval_<pose>.json`` next to the checkpoint).
This script gathers every eval JSON of the given run directories and reports
**three independent statuses** — schema presence, protocol semantics, and
safety disposition are never conflated:

- ``metadata_coverage`` (PASS/FAIL): every required top-level and per-rollout
  field is present (schema completeness only);
- ``protocol_consistency`` (PASS/FAIL): the primary D0–D4 files of one run
  agree semantically — same checkpoint (path + sha256), action space, obs
  preset, exact + source dataset fingerprints, split identity, success
  threshold/semantics, tick budget, control dt, policy sampler/horizon
  settings; rows agree with their file's top-level metadata; seeds follow the
  declared protocol with no duplicates; door poses match the pose plan;
  genuine repeat-same-seed determinism evidence exists and passed; and
  diagnostics stay out of the primary matrix;
- ``safety_readiness`` (PASS/REVIEW_REQUIRED/FAIL): machine-readable safety
  disposition with explicit reasons and counts. Unsafe/invalid adapter
  warnings or systematic rejections FAIL; unresolved force-admission-bound
  exceedances and env truncations produce at least REVIEW_REQUIRED and can
  never hide behind a metadata PASS.

Exit code is non-zero on any FAIL (pass ``--fail-on-review`` to also fail on
REVIEW_REQUIRED). Run through the official launcher (pure Python, no Kit)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/summarize_smoke_eval.py --out outputs/local_smoke_n50/summary.json \
        --pose-plan configs/door_pose_plan_v2_pose.json \
        --seed-plan configs/local_smoke_eval_plan_n50.json \
        outputs/door_push_alex_v2/act_door_push/local_smoke_act_a2_n50_seed0 [...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alexdoor_xas.policies.common.rollout_eval import trace_payload_hash

REQUIRED_ROW_FIELDS = (
    "seed",
    "randomized",
    "door_pose_id",
    "door_yaw_deg",
    "door_offset_xy",
    "success",
    "failure_label",
    "termination_reason",
    "first_success_tick",
    "time_to_success_s",
    "env_truncated",
    "start_pose_settle",
    "initial_angle_rad",
    "final_angle_rad",
    "n_ticks",
    "contact_ticks",
    "contact_source",
    "force_n",
    "force_trace_evidence",
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
    "checkpoint_sha256",
    "action_space",
    "obs_preset",
    "max_ticks",
    "success_angle_deg",
    "success_semantics",
    "base_seed",
    "door_pose",
    "control_dt",
    "dataset_provenance",
    "seed_protocol",
    "determinism_probe",
    "rollouts",
    "aggregate",
)
REQUIRED_PROVENANCE_FIELDS = (
    "source_fingerprint_sha256",
    "checkpoint_dataset_fingerprint_sha256",
    "live_dataset_fingerprint_sha256",
    "split_fingerprint_sha256",
    "checkpoint_split_fingerprint_sha256",
    "dataset_fingerprint_match",
    "split_fingerprint_match",
    "train_split_match",
    "val_split_match",
)
DIFFUSION_TOP_FIELDS = ("horizon", "n_action_steps", "sampler", "num_inference_steps")
DIFFUSION_ROW_FIELDS = ("sampler", "num_inference_steps")
ACT_TOP_FIELDS = ("chunk_size", "temporal_ensemble")
REQUIRED_PROBE_FIELDS = (
    "kind",
    "seed",
    "repeats",
    "tolerances",
    "trace_sha256",
    "reference_traces",
    "max_abs_diffs",
    "mismatches",
    "passed",
)
REQUIRED_TRACE_FIELDS = (
    "n_ticks",
    "requested",
    "applied",
    "statuses",
    "first_success_tick",
    "termination_reason",
    "final_angle_rad",
    "contact",
    "force",
)
REQUIRED_TOLERANCES = ("command_abs", "angle_abs_rad", "force_abs_n")
REQUIRED_MAX_DIFFS = ("requested", "applied", "final_angle_rad", "force_n")

# Fields that must be identical across every primary pose file of one run
# (same checkpoint, same dataset binding, same evaluation protocol).
RUN_CONSISTENT_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "action_space",
    "obs_preset",
    "max_ticks",
    "success_angle_deg",
    "success_semantics",
    "control_dt",
)
# Fields that must additionally be identical across *runs* (matched protocol).
MATRIX_CONSISTENT_FIELDS = (
    "max_ticks",
    "success_angle_deg",
    "success_semantics",
    "control_dt",
)
# Row-level fields that must equal the file's top-level policy metadata.
ROW_TOP_AGREEMENT = ("sampler", "num_inference_steps")

# Adapter warnings whose presence means invalid/unsafe execution, not a
# benign watch item (warn-level joint-velocity flags stay review-only).
UNSAFE_WARNING_MARKERS = ("non-finite", "invalid", "unsafe", "nan")
# A rejection storm is an adapter/frame problem; a stray rejection is a watch
# item. Threshold on the run's total rejected fraction of all commands.
SYSTEMATIC_REJECTION_FRACTION = 0.02


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
    parser.add_argument(
        "--pose-plan",
        type=Path,
        default=None,
        help="Pose-plan JSON; door poses in the eval files must match it.",
    )
    parser.add_argument(
        "--seed-plan",
        type=Path,
        default=Path("configs/local_smoke_eval_plan_n50.json"),
        help="Per-pose base seed and fixed/randomized counts for the primary matrix.",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Exit non-zero when safety_readiness is REVIEW_REQUIRED (default: FAIL only).",
    )
    return parser.parse_args()


def check_coverage(payload: dict, path: Path, problems: list[str]) -> None:
    """Schema completeness only: every required field is present."""
    is_diffusion = "diffusion" in path.name
    top_fields = REQUIRED_TOP_FIELDS + (DIFFUSION_TOP_FIELDS if is_diffusion else ACT_TOP_FIELDS)
    for field in top_fields:
        if field not in payload:
            problems.append(f"{path}: missing top-level field {field!r}")
    row_fields = REQUIRED_ROW_FIELDS + (DIFFUSION_ROW_FIELDS if is_diffusion else ())
    for i, row in enumerate(payload.get("rollouts", [])):
        missing = [field for field in row_fields if field not in row]
        if missing:
            problems.append(f"{path}: rollout {i} (seed {row.get('seed')}) missing {missing}")
    provenance = payload.get("dataset_provenance") or {}
    for field in REQUIRED_PROVENANCE_FIELDS:
        if not provenance.get(field):
            problems.append(f"{path}: dataset_provenance has no {field}")
    if payload.get("aggregate", {}).get("fixed_reset_spread_rad") is None and (
        payload.get("seed_protocol", {}).get("episodes_fixed", 0) > 0
    ):
        problems.append(f"{path}: fixed block present but no fixed_reset spread recorded")
    probe = payload.get("determinism_probe")
    if isinstance(probe, dict):
        for field in REQUIRED_PROBE_FIELDS:
            if field not in probe:
                problems.append(f"{path}: determinism_probe missing field {field!r}")
        reference = probe.get("reference_traces")
        if isinstance(reference, dict):
            for field in REQUIRED_TRACE_FIELDS:
                if field not in reference:
                    problems.append(
                        f"{path}: determinism_probe reference_traces missing field {field!r}"
                    )
        else:
            problems.append(f"{path}: determinism_probe reference_traces is not an object")


def _policy_metadata(payload: dict) -> dict:
    return {
        key: payload.get(key)
        for key in (
            "chunk_size",
            "temporal_ensemble",
            "horizon",
            "n_action_steps",
            "sampler",
            "num_inference_steps",
        )
        if key in payload
    }


def _provenance_identity(payload: dict) -> dict:
    provenance = payload.get("dataset_provenance") or {}
    return {key: provenance.get(key) for key in REQUIRED_PROVENANCE_FIELDS}


def _check_provenance_binding(payload: dict, path: Path, problems: list[str]) -> None:
    provenance = payload.get("dataset_provenance") or {}
    checkpoint_fp = provenance.get("checkpoint_dataset_fingerprint_sha256")
    live_fp = provenance.get("live_dataset_fingerprint_sha256")
    if checkpoint_fp != live_fp:
        problems.append(
            f"{path}: checkpoint/live dataset fingerprint mismatch — "
            f"{checkpoint_fp!r} != {live_fp!r}"
        )
    checkpoint_split_fp = provenance.get("checkpoint_split_fingerprint_sha256")
    live_split_fp = provenance.get("split_fingerprint_sha256")
    if checkpoint_split_fp != live_split_fp:
        problems.append(
            f"{path}: checkpoint/live split fingerprint mismatch — "
            f"{checkpoint_split_fp!r} != {live_split_fp!r}"
        )
    for field in (
        "dataset_fingerprint_match",
        "split_fingerprint_match",
        "train_split_match",
        "val_split_match",
    ):
        if provenance.get(field) is not True:
            problems.append(f"{path}: provenance match flag {field} is not true")


def check_file_protocol(
    payload: dict,
    path: Path,
    pose_plan_poses: dict[str, dict] | None,
    seed_plan_poses: dict[str, dict] | None,
    problems: list[str],
) -> None:
    """Within-file semantic consistency: rows vs top metadata, seeds, poses."""
    door_pose = payload.get("door_pose") or {}
    pose_id = door_pose.get("door_pose_id")
    rows = payload.get("rollouts", [])
    _check_provenance_binding(payload, path, problems)

    # Rows must agree with the file's top-level door pose and policy metadata.
    for i, row in enumerate(rows):
        for key, top_value in (
            ("door_pose_id", pose_id),
            ("door_yaw_deg", door_pose.get("door_yaw_deg")),
            ("door_offset_xy", door_pose.get("door_offset_xy")),
        ):
            if row.get(key) != top_value:
                problems.append(
                    f"{path}: rollout {i} {key}={row.get(key)!r} disagrees with "
                    f"top-level door_pose {top_value!r}"
                )
        for key in ROW_TOP_AGREEMENT:
            if key in payload and key in row and row[key] != payload[key]:
                problems.append(
                    f"{path}: rollout {i} {key}={row[key]!r} disagrees with "
                    f"top-level {payload[key]!r}"
                )
        if bool(row.get("success")) != (row.get("failure_label") is None):
            problems.append(
                f"{path}: rollout {i} success={row.get('success')} inconsistent with "
                f"failure_label={row.get('failure_label')!r}"
            )
        if row.get("force_exceeds_admission_bound"):
            evidence = row.get("force_trace_evidence") or {}
            force_max = (row.get("force_n") or {}).get("max")
            if (
                evidence.get("peak_force_n") != force_max
                or int(evidence.get("n_exceedance_ticks") or 0) < 1
                or not evidence.get("exceedance_ticks")
            ):
                problems.append(
                    f"{path}: rollout {i} force trace evidence does not bind the "
                    "reported admission-bound exceedance"
                )

    # Seed protocol: declared fixed/randomized seeds, exactly once each.
    protocol = payload.get("seed_protocol") or {}
    expected_fixed = list(protocol.get("fixed_seeds") or [])
    expected_randomized = list(protocol.get("randomized_seeds") or [])
    seeds = [row.get("seed") for row in rows]
    if sorted(set(seeds)) != sorted(seeds):
        duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
        problems.append(f"{path}: duplicate rollout seeds {duplicates}")
    fixed_seeds = [row.get("seed") for row in rows if not row.get("randomized")]
    randomized_seeds = [row.get("seed") for row in rows if row.get("randomized")]
    if fixed_seeds != expected_fixed or randomized_seeds != expected_randomized:
        problems.append(
            f"{path}: rollout seeds (fixed {fixed_seeds}, randomized {randomized_seeds}) "
            f"do not match the declared seed protocol (fixed {expected_fixed}, "
            f"randomized {expected_randomized})"
        )
    if seed_plan_poses is not None:
        expected_protocol = seed_plan_poses.get(pose_id)
        if expected_protocol is None:
            problems.append(f"{path}: pose {pose_id!r} is absent from the seed plan")
        else:
            for key in ("base_seed", "episodes_fixed", "episodes_randomized"):
                got = payload.get(key) if key == "base_seed" else protocol.get(key)
                if got != expected_protocol.get(key):
                    problems.append(
                        f"{path}: {key}={got!r} does not match seed plan "
                        f"{expected_protocol.get(key)!r} for pose {pose_id}"
                    )

    # Door pose must match the configured pose plan.
    if pose_plan_poses is not None and pose_id in pose_plan_poses:
        plan = pose_plan_poses[pose_id]
        expected = {
            "door_yaw_deg": plan.get("door_yaw_deg", 0.0),
            "door_offset_xy": [plan.get("door_offset_x_m", 0.0), plan.get("door_offset_y_m", 0.0)],
        }
        for key, value in expected.items():
            got = door_pose.get(key)
            matches = (
                all(abs(a - b) < 1e-9 for a, b in zip(got, value, strict=False))
                if isinstance(value, list) and isinstance(got, list) and len(got) == len(value)
                else got == value or (
                    isinstance(got, (int, float)) and isinstance(value, (int, float))
                    and abs(got - value) < 1e-9
                )
            )
            if not matches:
                problems.append(
                    f"{path}: door_pose {key}={got!r} does not match pose plan {value!r} "
                    f"for pose {pose_id}"
                )

    # Genuine repeat-same-seed determinism evidence, required per primary
    # file: the recorded first-fixed-rollout traces must have been reproduced
    # by at least one fresh-process replay (same reset seed, sampling seed,
    # pose, checkpoint, configuration). Within-process repeats are not
    # acceptable evidence (sim state is history-dependent per episode).
    probe = payload.get("determinism_probe")
    if not probe:
        problems.append(f"{path}: no repeat-same-seed determinism probe recorded")
    else:
        if probe.get("kind") != "repeat_same_seed_fresh_process":
            problems.append(f"{path}: determinism probe kind {probe.get('kind')!r} is not "
                            "repeat_same_seed_fresh_process")
        if int(probe.get("repeats") or 0) < 2:
            problems.append(
                f"{path}: determinism probe repeats={probe.get('repeats')} (< 2) — "
                "the fresh-process replay has not run"
            )
        if probe.get("passed") is not True:
            problems.append(
                f"{path}: determinism probe not passed: {probe.get('mismatches')}"
            )
        if probe.get("seed") != payload.get("base_seed"):
            problems.append(
                f"{path}: determinism probe seed {probe.get('seed')!r} does not match "
                f"base_seed {payload.get('base_seed')!r}"
            )
        repeats = int(probe.get("repeats") or 0)
        hashes = probe.get("trace_sha256")
        if not isinstance(hashes, list) or len(hashes) != repeats:
            problems.append(
                f"{path}: determinism probe trace_sha256 count "
                f"{len(hashes) if isinstance(hashes, list) else None} != repeats {repeats}"
            )
        elif any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            problems.append(f"{path}: determinism probe contains an invalid sha256 trace hash")
        reference = probe.get("reference_traces")
        if isinstance(reference, dict) and hashes:
            try:
                reference_hash = trace_payload_hash(reference)
            except (KeyError, TypeError, ValueError) as error:
                problems.append(
                    f"{path}: determinism probe reference trace is invalid: {error}"
                )
            else:
                if hashes[0] != reference_hash:
                    problems.append(
                        f"{path}: determinism probe first trace hash does not match "
                        "reference_traces"
                    )
                for index, replay_hash in enumerate(hashes[1:], start=1):
                    if replay_hash != reference_hash:
                        problems.append(
                            f"{path}: determinism replay trace hash {index} differs "
                            "from reference_traces"
                        )
        tolerances = probe.get("tolerances")
        for key in REQUIRED_TOLERANCES:
            value = tolerances.get(key) if isinstance(tolerances, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                problems.append(
                    f"{path}: determinism probe tolerance {key!r} is not finite/non-negative"
                )
        max_diffs = probe.get("max_abs_diffs")
        for key in REQUIRED_MAX_DIFFS:
            value = max_diffs.get(key) if isinstance(max_diffs, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                problems.append(
                    f"{path}: determinism probe max_abs_diffs {key!r} is invalid"
                )
        if isinstance(tolerances, dict) and isinstance(max_diffs, dict):
            for diff_key, tolerance_key in (
                ("requested", "command_abs"),
                ("applied", "command_abs"),
                ("final_angle_rad", "angle_abs_rad"),
                ("force_n", "force_abs_n"),
            ):
                diff = max_diffs.get(diff_key)
                tolerance = tolerances.get(tolerance_key)
                if (
                    isinstance(diff, (int, float))
                    and math.isfinite(diff)
                    and isinstance(tolerance, (int, float))
                    and math.isfinite(tolerance)
                    and diff > tolerance
                ):
                    problems.append(
                        f"{path}: determinism {diff_key} max difference {diff} "
                        f"exceeds stored tolerance {tolerance_key}={tolerance}"
                    )
        mismatches = probe.get("mismatches")
        if not isinstance(mismatches, list):
            problems.append(f"{path}: determinism probe mismatches is not a list")
        elif probe.get("passed") is not (len(mismatches) == 0):
            problems.append(
                f"{path}: determinism probe passed={probe.get('passed')!r} disagrees "
                f"with {len(mismatches)} mismatch(es)"
            )


def check_run_consistency(
    primary: dict[str, tuple[Path, dict]], run_dir: Path, problems: list[str]
) -> None:
    """All primary pose files of one run must describe one checkpoint/protocol."""
    if not primary:
        return
    reference_pose = sorted(primary)[0]
    ref_path, reference = primary[reference_pose]
    for pose_id in sorted(primary):
        path, payload = primary[pose_id]
        for field in RUN_CONSISTENT_FIELDS:
            if payload.get(field) != reference.get(field):
                problems.append(
                    f"{run_dir}: {field} mismatch across pose files — "
                    f"{path.name}={payload.get(field)!r} vs {ref_path.name}="
                    f"{reference.get(field)!r}"
                )
        if _policy_metadata(payload) != _policy_metadata(reference):
            problems.append(
                f"{run_dir}: policy metadata mismatch across pose files — "
                f"{path.name}={_policy_metadata(payload)} vs {ref_path.name}="
                f"{_policy_metadata(reference)}"
            )
        if _provenance_identity(payload) != _provenance_identity(reference):
            problems.append(
                f"{run_dir}: dataset/split identity mismatch across pose files — "
                f"{path.name}={_provenance_identity(payload)} vs {ref_path.name}="
                f"{_provenance_identity(reference)}"
            )


def assess_safety(run_name: str, rows: list[dict]) -> dict[str, Any]:
    """Safety/readiness disposition for one run: PASS / REVIEW_REQUIRED / FAIL."""
    fail_reasons: list[str] = []
    review_reasons: list[str] = []

    non_finite_rows: list[int | None] = []
    for row in rows:
        force = row.get("force_n")
        values = [] if force is None else [force.get(key) for key in ("mean", "max", "p95")]
        values.append(row.get("impulse_ns"))
        if any(
            value is not None
            and (not isinstance(value, (int, float)) or not math.isfinite(value))
            for value in values
        ):
            non_finite_rows.append(row.get("seed"))
    if non_finite_rows:
        fail_reasons.append(
            f"non-finite rollout force/contact evidence (seeds {non_finite_rows})"
        )

    n_commands = sum(
        int(row.get("n_accepted", 0)) + int(row.get("n_corrected", 0))
        + int(row.get("n_rejected", 0))
        for row in rows
    )
    n_rejected = sum(int(row.get("n_rejected", 0)) for row in rows)
    rejection_labels = [
        row.get("seed")
        for row in rows
        if row.get("failure_label") in ("stopped_on_rejection", "commands_rejected")
    ]
    if n_commands and n_rejected / n_commands >= SYSTEMATIC_REJECTION_FRACTION:
        fail_reasons.append(
            f"systematic adapter rejections: {n_rejected}/{n_commands} commands rejected"
        )
    elif rejection_labels:
        fail_reasons.append(
            f"rollouts failed on adapter rejections (seeds {rejection_labels})"
        )
    elif n_rejected:
        review_reasons.append(f"{n_rejected} adapter rejection(s) (below systematic threshold)")

    unsafe_warnings: dict[str, int] = {}
    for row in rows:
        for message, count in (row.get("warning_counts") or {}).items():
            if any(marker in message.lower() for marker in UNSAFE_WARNING_MARKERS):
                unsafe_warnings[message] = unsafe_warnings.get(message, 0) + int(count)
    if unsafe_warnings:
        fail_reasons.append(f"unsafe/invalid adapter warnings: {unsafe_warnings}")

    force_rows = [row for row in rows if row.get("force_exceeds_admission_bound")]
    if force_rows:
        peak = max(
            (row.get("force_n") or {}).get("max") or 0.0 for row in force_rows
        )
        review_reasons.append(
            f"{len(force_rows)} rollout(s) exceed the 200 N dataset-admission force "
            f"bound (peak {peak:.1f} N; seeds "
            f"{[row.get('seed') for row in force_rows]})"
        )

    truncated = [row.get("seed") for row in rows if row.get("env_truncated")]
    if truncated:
        review_reasons.append(f"env truncation hit on seeds {truncated}")

    n_warnings = sum(int(row.get("n_warnings", 0)) for row in rows)
    status = "FAIL" if fail_reasons else ("REVIEW_REQUIRED" if review_reasons else "PASS")
    return {
        "status": status,
        "fail_reasons": fail_reasons,
        "review_reasons": review_reasons,
        "counts": {
            "n_rollouts": len(rows),
            "n_commands": n_commands,
            "n_rejected": n_rejected,
            "n_warnings": n_warnings,
            "n_force_exceeds_admission_bound": len(force_rows),
            "n_env_truncated": len(truncated),
        },
        "run": run_name,
    }


def summarize(
    run_dirs: list[Path],
    expected_poses: set[str],
    pose_plan: dict | None,
    seed_plan: dict | None = None,
) -> dict[str, Any]:
    """Build the full summary (pure; no exit-code policy)."""
    coverage_problems: list[str] = []
    protocol_problems: list[str] = []
    runs: dict[str, dict] = {}
    matrix_reference: tuple[Path, dict] | None = None
    pose_plan_poses = (
        {pose["pose_id"]: pose for pose in pose_plan["poses"]} if pose_plan else None
    )
    seed_plan_poses = seed_plan.get("poses") if seed_plan else None

    for run_dir in run_dirs:
        # Pose-qualified files only: a stale legacy <policy>_eval.json (no pose
        # id) must never be ingested as a phantom default-pose block.
        eval_files = sorted((run_dir / "metrics").glob("*_eval_*.json"))
        if not eval_files:
            coverage_problems.append(f"{run_dir}: no eval JSONs under metrics/")
            continue
        poses: dict[str, dict] = {}
        primary: dict[str, tuple[Path, dict]] = {}
        diagnostics: dict[str, dict] = {}
        rows_all: list[dict] = []
        for path in eval_files:
            payload = json.loads(path.read_text())
            check_coverage(payload, path, coverage_problems)
            pose_id = (payload.get("door_pose") or {}).get("door_pose_id", "D0")
            if pose_id in poses or pose_id in diagnostics:
                protocol_problems.append(f"{run_dir}: duplicate eval files for pose {pose_id!r}")
                continue
            entry = {
                "eval_json": str(path),
                "door_pose": payload.get("door_pose"),
                "base_seed": payload.get("base_seed"),
                "sampler": payload.get("sampler"),
                "num_inference_steps": payload.get("num_inference_steps"),
                "determinism_probe_passed": (payload.get("determinism_probe") or {}).get(
                    "passed"
                ),
                "aggregate": payload.get("aggregate"),
            }
            if "diag" in pose_id:
                # Diagnostics (e.g. the DDPM-100 hard-seed probe) are reported
                # separately and never enter the main matrix aggregates.
                diagnostics[pose_id] = entry
                continue
            check_file_protocol(
                payload, path, pose_plan_poses, seed_plan_poses, protocol_problems
            )
            poses[pose_id] = entry
            primary[pose_id] = (path, payload)
            rows_all.extend(payload.get("rollouts", []))
        if set(poses) != expected_poses:
            protocol_problems.append(
                f"{run_dir}: pose matrix incomplete — got {sorted(poses)}, "
                f"expected {sorted(expected_poses)}"
            )
        check_run_consistency(primary, run_dir, protocol_problems)

        # Matched protocol across runs (one evaluation contract per matrix).
        if primary:
            ref_pose = sorted(primary)[0]
            path, payload = primary[ref_pose]
            if matrix_reference is None:
                matrix_reference = (path, payload)
            else:
                ref_path, reference = matrix_reference
                for field in MATRIX_CONSISTENT_FIELDS:
                    if payload.get(field) != reference.get(field):
                        protocol_problems.append(
                            f"matrix: {field} mismatch across runs — "
                            f"{path}={payload.get(field)!r} vs {ref_path}="
                            f"{reference.get(field)!r}"
                        )

        n_success = sum(1 for row in rows_all if row.get("success"))
        sample = primary[sorted(primary)[0]][1] if primary else json.loads(
            eval_files[0].read_text()
        )
        runs[run_dir.name] = {
            "run_dir": str(run_dir),
            "checkpoint": sample.get("checkpoint"),
            "checkpoint_sha256": sample.get("checkpoint_sha256"),
            "action_space": sample.get("action_space"),
            "obs_preset": sample.get("obs_preset"),
            "success_semantics": sample.get("success_semantics"),
            "policy_metadata": _policy_metadata(sample),
            "dataset_identity": _provenance_identity(sample),
            "poses": dict(sorted(poses.items())),
            "diagnostics": dict(sorted(diagnostics.items())),
            "safety_readiness": assess_safety(run_dir.name, rows_all),
            "overall": {
                "n_rollouts": len(rows_all),
                "n_success": n_success,
                "success_rate": (n_success / len(rows_all)) if rows_all else None,
                "termination_reasons": sorted(
                    {
                        row["termination_reason"]
                        for row in rows_all
                        if row.get("termination_reason")
                    }
                ),
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
                "peak_force_n": max(
                    (
                        (row.get("force_n") or {}).get("max") or 0.0
                        for row in rows_all
                    ),
                    default=None,
                ),
            },
        }

    safety_statuses = [run["safety_readiness"]["status"] for run in runs.values()]
    overall_safety = (
        "FAIL"
        if "FAIL" in safety_statuses or not safety_statuses
        else ("REVIEW_REQUIRED" if "REVIEW_REQUIRED" in safety_statuses else "PASS")
    )
    return {
        "schema": "alexdoor_xas.smoke_eval_summary.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "metadata_coverage": "PASS" if not coverage_problems else "FAIL",
        "coverage_problems": coverage_problems,
        "protocol_consistency": "PASS" if not protocol_problems else "FAIL",
        "protocol_problems": protocol_problems,
        "safety_readiness": overall_safety,
        "runs": runs,
    }


def main() -> int:
    args = parse_args()
    expected_poses = {pose.strip() for pose in args.expected_poses.split(",") if pose.strip()}
    pose_plan = json.loads(args.pose_plan.read_text()) if args.pose_plan else None
    seed_plan = json.loads(args.seed_plan.read_text()) if args.seed_plan else None
    summary = summarize(args.run_dirs, expected_poses, pose_plan, seed_plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    for name, run in summary["runs"].items():
        overall = run["overall"]
        safety = run["safety_readiness"]
        print(
            f"[{name}] rollouts={overall['n_rollouts']} "
            f"success={overall['n_success']}/{overall['n_rollouts']} "
            f"rejected={overall['adapter']['n_rejected']} "
            f"warnings={overall['adapter']['n_warnings']} "
            f"force_exceedances={overall['n_force_exceeds_admission_bound']} "
            f"safety={safety['status']} poses={list(run['poses'])}",
            flush=True,
        )
        for reason in safety["fail_reasons"]:
            print(f"  [safety-fail] {reason}", flush=True)
        for reason in safety["review_reasons"]:
            print(f"  [safety-review] {reason}", flush=True)

    failed = False
    for label, status_key, problems_key in (
        ("metadata coverage", "metadata_coverage", "coverage_problems"),
        ("protocol consistency", "protocol_consistency", "protocol_problems"),
    ):
        if summary[status_key] == "FAIL":
            failed = True
            for problem in summary[problems_key]:
                print(f"[fail] {problem}", flush=True)
            print(f"FAIL: {label} incomplete ({args.out})", flush=True)
    if summary["safety_readiness"] == "FAIL":
        failed = True
        print(f"FAIL: safety readiness ({args.out})", flush=True)
    elif summary["safety_readiness"] == "REVIEW_REQUIRED":
        print(f"REVIEW_REQUIRED: safety readiness ({args.out})", flush=True)
        if args.fail_on_review:
            failed = True
    if failed:
        return 1
    print(
        f"PASS: metadata coverage + protocol consistency; safety "
        f"{summary['safety_readiness']} ({args.out})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
