#!/usr/bin/env python
"""Merge per-pose generation runs into one official dataset version (no Kit).

The door-task pose is fixed per Isaac process, so a multi-pose dataset is
generated as one ``run_scripted_baseline.py --no-export`` run per pose (each
records episodes under ``outputs/<experiment>/pose<id>/`` only). This script
is the single writer of the official version directory: it reloads every
episode, verifies the merged set against the pose plan (episode counts, unique
ids, disjoint seed blocks, pose coverage, per-run sanity summaries, task
success), exports all action spaces once via the frozen export path, and
writes a contract-grade ``manifest.json`` next to each ``meta.json``.

Run through the official launcher (pure Python, no Kit)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/export_merged_dataset.py --pose-plan configs/door_pose_plan_v2_pose.json \
        --experiment alex_v2_n50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.data_engine.export import export_datasets
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from alexdoor_xas.policies.scripted import DoorPushControllerCfg
from alexdoor_xas.recording import EpisodeBuffer, read_episode

TASK_INSTRUCTION = "push the door open"
"""Language placeholder (constant for now): every episode of this task family
carries the same instruction until language variation becomes a research axis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-plan",
        type=Path,
        required=True,
        help="Pose-plan JSON (configs/door_pose_plan_*.json) the runs were generated from.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="outputs/<experiment>/ holding one pose<id> run dir per plan pose.",
    )
    parser.add_argument("--outputs-root", type=Path, default=paths.OUTPUTS_DIR)
    parser.add_argument("--datasets-root", type=Path, default=paths.DATASETS_DIR)
    return parser.parse_args()


def _load_run_dir(run_dir: Path, pose_id: str) -> tuple[list[Path], list[EpisodeBuffer], dict]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(
            f"pose {pose_id}: no episodes at {episodes_dir} — "
            "generate every pose run before merging (partial merges are refused)"
        )
    sanity_path = run_dir / "metrics" / "sanity.json"
    if not sanity_path.is_file():
        raise FileNotFoundError(f"pose {pose_id}: missing sanity summary {sanity_path}")
    sanity = json.loads(sanity_path.read_text())
    if sanity["n_episodes_with_errors"]:
        raise RuntimeError(
            f"pose {pose_id}: {sanity['n_episodes_with_errors']} episode(s) with "
            f"sanity errors — refusing to merge (see {sanity_path})"
        )
    episode_files = sorted(episodes_dir.glob("episode_*.hdf5"))
    return episode_files, [read_episode(path) for path in episode_files], sanity


def load_pose_runs(plan: dict, outputs_root: Path, experiment: str) -> dict[str, dict]:
    """Load and verify each pose run (+ optional overdraw run): sanity-clean only."""
    runs: dict[str, dict] = {}
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        run_dir = outputs_root / experiment / f"pose{pose_id}"
        episode_files, episodes, sanity = _load_run_dir(run_dir, pose_id)
        overdraw_dir = outputs_root / experiment / f"pose{pose_id}_overdraw"
        overdraw_files: list[Path] = []
        overdraw_episodes: list[EpisodeBuffer] = []
        if overdraw_dir.is_dir():
            overdraw_files, overdraw_episodes, _ = _load_run_dir(
                overdraw_dir, f"{pose_id} (overdraw)"
            )
        runs[pose_id] = {
            "pose": pose,
            "run_dir": run_dir,
            "episode_files": episode_files,
            "episodes": episodes,
            "overdraw_files": overdraw_files,
            "overdraw_episodes": overdraw_episodes,
            "sanity": sanity,
        }
    return runs


def verify_merged(plan: dict, runs: dict[str, dict]) -> tuple[list[EpisodeBuffer], list[dict]]:
    """Select and verify the official episode set per the plan's overdraw policy.

    Per pose: successful episodes from the primary seed block in seed order;
    task-failed draws are skipped (reported, never hidden) and replaced by
    successful overdraw-run episodes in seed order until the pose count is
    reached. Any sanity error already refused the merge at load time. Returns
    ``(merged_episodes, skipped_report)``.
    """
    block = int(plan["seed_block_size"])
    n_per_pose = int(plan["episodes_fixed_per_pose"]) + int(plan["episodes_randomized_per_pose"])
    merged: list[EpisodeBuffer] = []
    skipped: list[dict] = []
    problems: list[str] = []
    for pose_id, run in runs.items():
        pose = run["pose"]
        base = int(pose["base_seed"])
        if base % block:
            problems.append(f"pose {pose_id}: base seed {base} not block-aligned ({block})")
        expected_seeds = list(range(base, base + n_per_pose))
        seeds = sorted(episode.meta.seed for episode in run["episodes"])
        if seeds != expected_seeds:
            problems.append(f"pose {pose_id}: seeds {seeds} != expected block {expected_seeds}")

        overdraw_base = int(pose.get("overdraw_base_seed", -1))
        for episode in run["overdraw_episodes"]:
            if not (0 <= episode.meta.seed - overdraw_base < block):
                problems.append(
                    f"pose {pose_id}: overdraw seed {episode.meta.seed} outside "
                    f"namespace [{overdraw_base}, {overdraw_base + block})"
                )

        selected: list[EpisodeBuffer] = []
        for episode in sorted(
            run["episodes"] + run["overdraw_episodes"], key=lambda e: e.meta.seed
        ):
            sanity = check_alex_episode(
                episode, force_error_n=FORCE_DATASET_LIMIT_N
            )
            if sanity.errors:
                problems.append(
                    f"pose {pose_id} seed {episode.meta.seed}: current dataset safety gate "
                    + "; ".join(sanity.errors)
                )
                continue
            recorded_pose = episode.extras.get("door_pose_id")
            if recorded_pose != pose_id:
                problems.append(
                    f"pose {pose_id} seed {episode.meta.seed}: episode records "
                    f"door_pose_id={recorded_pose!r}"
                )
                continue
            if episode.outcome is None or not episode.outcome.success:
                skipped.append(
                    {
                        "pose_id": pose_id,
                        "seed": episode.meta.seed,
                        "failure_label": episode.outcome.failure_label
                        if episode.outcome
                        else "missing outcome",
                        "reason": "task failure — replaced per the plan's overdraw policy",
                    }
                )
                continue
            if "door_yaw_rad" not in episode.steps[0].object_state:
                problems.append(
                    f"pose {pose_id} seed {episode.meta.seed}: missing door-pose obs terms"
                )
                continue
            if len(selected) < n_per_pose:
                selected.append(episode)
        if len(selected) != n_per_pose:
            problems.append(
                f"pose {pose_id}: only {len(selected)} clean successful episodes "
                f"(need {n_per_pose}) — extend the overdraw run"
            )
        merged.extend(selected)

    ids = [episode.meta.episode_id for episode in merged]
    if len(set(ids)) != len(ids):
        problems.append("duplicate episode ids across pose runs")
    expected_total = n_per_pose * len(plan["poses"])
    if len(merged) != expected_total:
        problems.append(f"{len(merged)} episodes merged, expected {expected_total}")
    if problems:
        raise RuntimeError("merge verification failed:\n  - " + "\n  - ".join(problems))
    return merged, skipped


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def episode_contact_safety(episode: EpisodeBuffer) -> dict:
    """Per-episode contact/force/adapter/clamp summary for the manifest."""
    control_dt = float(episode.meta.control_dt)
    sensed = np.array([bool(step.contact.get("sensed")) for step in episode.steps])
    forces = np.array([float(step.contact.get("force_n", 0.0)) for step in episode.steps])
    contact_forces = forces[sensed] if sensed.any() else np.zeros(0)
    force_diagnostics = contact_force_diagnostics(
        episode, force_limit_n=FORCE_DATASET_LIMIT_N
    )
    return {
        "episode_id": episode.meta.episode_id,
        "seed": episode.meta.seed,
        "door_pose_id": episode.extras.get("door_pose_id"),
        "success": bool(episode.outcome.success) if episode.outcome else None,
        "failure_label": episode.outcome.failure_label if episode.outcome else None,
        "n_steps": episode.n_steps,
        "contact_ticks": int(sensed.sum()),
        "contact_source": episode.steps[0].contact.get("source") if episode.steps else None,
        "force_n": {
            "mean": float(contact_forces.mean()) if contact_forces.size else 0.0,
            "max": float(contact_forces.max()) if contact_forces.size else 0.0,
            "p95": _percentile(contact_forces, 95.0),
        },
        "force_admission": {
            "limit_n": FORCE_DATASET_LIMIT_N,
            "passed": force_diagnostics["force_admission_passed"],
            "all_forces_finite": force_diagnostics["all_forces_finite"],
            "non_finite_force_ticks": force_diagnostics["non_finite_force_ticks"],
            "min_force_n": force_diagnostics["min_force_n"],
            "min_force_tick": force_diagnostics["min_force_tick"],
            "negative_force_ticks": force_diagnostics["negative_force_ticks"],
            "max_force_n": force_diagnostics["max_force_n"],
            "ticks_over_limit": force_diagnostics["ticks_over_limit"],
            "terminal": force_diagnostics["terminal"],
        },
        "impulse_ns": float((forces * control_dt).sum()),
        "safety_clamp_flags": {
            "pos_clamped_ticks": int(
                sum(bool(step.safety.get("pos_clamped")) for step in episode.steps)
            ),
            "rot_clamped_ticks": int(
                sum(bool(step.safety.get("rot_clamped")) for step in episode.steps)
            ),
        },
        "ik_clamp_telemetry": episode.extras.get("ik_clamp_telemetry"),
    }


def pose_object_metadata(run: dict) -> dict:
    """Object/frame metadata per pose (hinge axis, frames, contact target)."""
    episode = run["episodes"][0]
    controller_cfg = DoorPushControllerCfg(**episode.extras["controller_cfg"])
    return {
        "pose_id": run["pose"]["pose_id"],
        "door_frame_pos_w": np.asarray(episode.extras["door_frame_pos_w"]).tolist(),
        "door_frame_quat_w_xyzw": np.asarray(
            episode.extras["door_frame_quat_w_xyzw"]
        ).tolist(),
        "hinge_axis": "door-frame +Z (frame origin at the Doorframe body)",
        "panel_frame": "door frame rotated by the hinge angle about +Z (action/frames.py)",
        "intended_contact_target_panel": [
            controller_cfg.surface_x_m(controller_cfg.contact_clearance_m),
            controller_cfg.push_point_y_m,
            controller_cfg.push_height_m,
        ],
        "scene_usd": episode.extras["engine_cfg"].get("scene"),
    }


def dataset_fingerprint(episode_files: list[Path]) -> str:
    """sha256 over the sorted per-file sha256s of the source episode HDF5s."""
    digest = hashlib.sha256()
    for path in sorted(episode_files):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def build_manifest(
    plan: dict,
    runs: dict[str, dict],
    merged: list[EpisodeBuffer],
    skipped: list[dict],
    fingerprint: str,
) -> dict:
    return {
        "schema": "alexdoor_xas.dataset_manifest.v1",
        "version": plan["version"],
        "task_instruction": TASK_INSTRUCTION,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_fingerprint_sha256": fingerprint,
        "force_admission_policy": {
            "min_contact_force_n": 0.0,
            "max_contact_force_n": FORCE_DATASET_LIMIT_N,
            "comparison": "every recorded control-tick sample is finite and within [0, limit]",
            "enforced_before_generation_export": True,
            "rechecked_before_merged_export": True,
        },
        "pose_plan": plan,
        "source_runs": {
            pose_id: str(run["run_dir"]) for pose_id, run in runs.items()
        },
        "object_metadata": [pose_object_metadata(run) for run in runs.values()],
        "episodes": [episode_contact_safety(episode) for episode in merged],
        "skipped_episodes": skipped,
        "sanity_summaries": {
            pose_id: {
                "n_episodes_checked": run["sanity"]["n_episodes_checked"],
                "n_episodes_with_errors": run["sanity"]["n_episodes_with_errors"],
                "n_episodes_with_warnings": run["sanity"]["n_episodes_with_warnings"],
            }
            for pose_id, run in runs.items()
        },
        # Typed slots for later phases: RGB capture and camera calibration can
        # land without a manifest schema break (empty/None until then).
        "cameras": [],
        "calibration": None,
    }


def main() -> int:
    args = parse_args()
    plan = json.loads(args.pose_plan.read_text())
    version = plan["version"]

    runs = load_pose_runs(plan, args.outputs_root, args.experiment)
    merged, skipped = verify_merged(plan, runs)
    merged_ids = {episode.meta.episode_id for episode in merged}
    selected_files = [
        path
        for run in runs.values()
        for path, episode in zip(
            run["episode_files"] + run["overdraw_files"],
            run["episodes"] + run["overdraw_episodes"],
            strict=True,
        )
        if episode.meta.episode_id in merged_ids
    ]
    fingerprint = dataset_fingerprint(selected_files)

    exported = export_datasets(merged, args.datasets_root, version=version)
    manifest = build_manifest(plan, runs, merged, skipped, fingerprint)
    for action_space, directory in exported.items():
        (Path(directory) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[export] {action_space} -> {directory}", flush=True)

    for entry in skipped:
        print(
            f"[skipped] pose {entry['pose_id']} seed {entry['seed']}: "
            f"{entry['failure_label']} ({entry['reason']})",
            flush=True,
        )
    print(
        f"[merge] {len(merged)} episodes from {len(runs)} poses -> version {version} "
        f"(fingerprint {fingerprint[:16]}...)",
        flush=True,
    )
    print("DONE: merged dataset export complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
