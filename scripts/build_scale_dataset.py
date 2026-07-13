#!/usr/bin/env python
"""Generate, atomically publish, and verify the v3 nested-scale master dataset.

Generation launches one isolated simulator process per door pose. Candidate
attempts and failures remain under outputs; only 110 clean, successful,
content-distinct episodes per pose can enter the paired A2/A3 master.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.cluster_sweep.config import SweepConfig, load_sweep_config
from alexdoor_xas.data_engine import export_paired_ee_datasets_atomic
from alexdoor_xas.dataset import (
    EpisodeDataset,
    build_nested_views,
    compute_norm_stats,
    dataset_fingerprint,
    episode_content_key,
    load_episode_record,
    load_norm_stats,
    load_view_payload,
    save_norm_stats,
    save_view_payload,
    split_entries,
    validate_nested_views,
    validate_norm_stats,
    view_norm_stats_path,
    view_path,
)
from alexdoor_xas.dataset.robot_asset import dataset_robot_asset_payload
from alexdoor_xas.eval.sanity import FORCE_DATASET_LIMIT_N, check_alex_episode
from alexdoor_xas.recording import EpisodeBuffer, read_episode

STATE_SCHEMA = "alexdoor_xas.scale_generation_state.v1"
MASTER_SCHEMA = "alexdoor_xas.scale_master_manifest.v1"
PUBLICATION_SCHEMA = "alexdoor_xas.scale_publication.v1"
DEFAULT_CONFIG = Path("configs/cluster_sweep.v1.json")
DEFAULT_PLAN = Path("configs/door_pose_plan_v3_scale.json")
DEFAULT_EXPERIMENT = "v3_scale_generation"
LAUNCHER = Path("/home/pacquadr/IsaacLab/isaaclab.sh")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pose-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--outputs-root", type=Path, default=paths.OUTPUTS_DIR)
    parser.add_argument("--datasets-root", type=Path, default=paths.DATASETS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sub.add_parser("publish")
    sub.add_parser("verify")
    return parser


def _load_plan(path: Path, config: SweepConfig) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    expected_root = {
        "schema",
        "version",
        "description",
        "calibration_fingerprint",
        "source_candidates_per_pose",
        "overdraw_candidates_per_pose",
        "selected_episodes_per_pose",
        "fixed_candidates_per_pose",
        "randomized_candidates_only",
        "selection_seed",
        "selection_algorithm",
        "force_admission_policy",
        "poses",
    }
    if not isinstance(plan, dict) or set(plan) != expected_root:
        raise ValueError("scale pose-plan root keys mismatch")
    if plan["schema"] != "alexdoor_xas.scale_pose_plan.v1":
        raise ValueError("scale pose-plan schema mismatch")
    if plan["version"] != config.dataset.master_version:
        raise ValueError("scale pose-plan version mismatch")
    if plan["source_candidates_per_pose"] != 110:
        raise ValueError("scale plan must contain 110 source candidates per pose")
    if plan["selected_episodes_per_pose"] != 110:
        raise ValueError("scale plan must select 110 episodes per pose")
    if plan["fixed_candidates_per_pose"] != 0 or plan["randomized_candidates_only"] is not True:
        raise ValueError("scale source candidates must all be randomized")
    if plan["selection_seed"] != config.selection.seed:
        raise ValueError("scale selection seed mismatch")
    if plan["selection_algorithm"] != config.selection.algorithm:
        raise ValueError("scale selection algorithm mismatch")
    if tuple(pose["pose_id"] for pose in plan["poses"]) != config.dataset.pose_ids:
        raise ValueError("scale pose inventory/order mismatch")
    authoritative_path = paths.REPO_ROOT / config.selection.pose_plan
    authoritative = json.loads(authoritative_path.read_text())
    calibration = json.loads((paths.REPO_ROOT / config.selection.calibration).read_text())
    if plan["calibration_fingerprint"] != calibration.get("fingerprint"):
        raise ValueError("scale calibration fingerprint differs from the canonical artifact")
    canonical = json.loads(
        (paths.REPO_ROOT / config.selection.canonical_pose_plan).read_text()
    )
    canonical_poses = {pose["pose_id"]: pose for pose in canonical["poses"]}
    authoritative_poses = {pose["pose_id"]: pose for pose in authoritative["poses"]}
    all_seeds: list[int] = []
    for pose in plan["poses"]:
        expected_pose_keys = {
            "pose_id",
            "door_yaw_deg",
            "door_yaw_rad",
            "door_offset_x_m",
            "door_offset_y_m",
            "source_seed_start",
            "source_seed_stop",
            "overdraw_seed_start",
            "overdraw_seed_stop",
            "validated_probe",
        }
        if set(pose) != expected_pose_keys:
            raise ValueError(f"pose {pose.get('pose_id')} plan keys mismatch")
        pose_id = pose["pose_id"]
        geometry_fields = (
            "door_yaw_deg",
            "door_yaw_rad",
            "door_offset_x_m",
            "door_offset_y_m",
        )
        if any(pose[name] != canonical_poses[pose_id][name] for name in geometry_fields):
            raise ValueError(f"pose {pose_id} geometry differs from the canonical pose definition")
        seed_fields = (
            "source_seed_start",
            "source_seed_stop",
            "overdraw_seed_start",
            "overdraw_seed_stop",
        )
        if any(pose[name] != authoritative_poses[pose_id][name] for name in seed_fields):
            raise ValueError(f"pose {pose_id} seed ranges differ from the authoritative plan")
        source = list(range(pose["source_seed_start"], pose["source_seed_stop"]))
        overdraw = list(range(pose["overdraw_seed_start"], pose["overdraw_seed_stop"]))
        if len(source) != plan["source_candidates_per_pose"]:
            raise ValueError(f"pose {pose['pose_id']} source seed namespace has wrong size")
        if len(overdraw) != plan["overdraw_candidates_per_pose"]:
            raise ValueError(f"pose {pose['pose_id']} overdraw namespace has wrong size")
        if set(source) & set(overdraw):
            raise ValueError(f"pose {pose['pose_id']} source/overdraw namespaces overlap")
        all_seeds.extend(source + overdraw)
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("scale seed namespaces overlap across poses")
    return plan


def _git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": commit,
        "clean_tree": not bool(status),
        "commit_time": _git("show", "-s", "--format=%cI", "HEAD"),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=paths.REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def _plan_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_path(outputs_root: Path, experiment: str) -> Path:
    return outputs_root / experiment / "generation_state.json"


def _load_or_create_state(
    outputs_root: Path,
    experiment: str,
    plan_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    state_path = _state_path(outputs_root, experiment)
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("generation state schema mismatch")
        if state.get("pose_plan_sha256") != _plan_hash(plan_path):
            raise ValueError("generation state belongs to a different pose plan")
        if state.get("source_git_commit") != source["commit"]:
            raise ValueError("generation state belongs to a different source commit")
        return state
    return {
        "schema": STATE_SCHEMA,
        "pose_plan": str(plan_path),
        "pose_plan_sha256": _plan_hash(plan_path),
        "source_git_commit": source["commit"],
        "poses": {},
    }


def _pose_seeds(pose: dict[str, Any]) -> list[int]:
    return [
        *range(pose["source_seed_start"], pose["source_seed_stop"]),
        *range(pose["overdraw_seed_start"], pose["overdraw_seed_stop"]),
    ]


def _verify_candidate_run(run_dir: Path, pose: dict[str, Any]) -> None:
    files = sorted((run_dir / "episodes").glob("episode_*.hdf5"))
    expected = _pose_seeds(pose)
    if len(files) != len(expected):
        raise ValueError(
            f"pose {pose['pose_id']} candidate run has {len(files)} episodes, "
            f"expected {len(expected)}"
        )
    episodes = [read_episode(path) for path in files]
    seeds = sorted(episode.meta.seed for episode in episodes)
    if seeds != sorted(expected):
        raise ValueError(f"pose {pose['pose_id']} candidate seed inventory mismatch")
    if any(episode.extras.get("door_pose_id") != pose["pose_id"] for episode in episodes):
        raise ValueError(f"pose {pose['pose_id']} candidate records wrong pose provenance")
    sanity_path = run_dir / "metrics" / "sanity.json"
    if not sanity_path.is_file():
        raise ValueError(f"pose {pose['pose_id']} candidate run has no sanity evidence")
    sanity = json.loads(sanity_path.read_text())
    if sanity.get("n_episodes_checked") != len(expected):
        raise ValueError(f"pose {pose['pose_id']} sanity candidate count mismatch")


def generate(
    config: SweepConfig,
    plan: dict[str, Any],
    *,
    plan_path: Path,
    outputs_root: Path,
    experiment: str,
) -> None:
    source = _git_state()
    if not source["clean_tree"]:
        raise RuntimeError("scale generation requires a clean committed checkout")
    state = _load_or_create_state(outputs_root, experiment, plan_path, source)
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        record = state["poses"].setdefault(pose_id, {"attempts": [], "completed": None})
        if record["completed"] is not None:
            completed = Path(record["completed"])
            _verify_candidate_run(completed, pose)
            print(f"[resume] {pose_id}: verified existing completed run {completed}")
            continue
        attempt_number = len(record["attempts"]) + 1
        run_id = f"pose{pose_id}_attempt{attempt_number:03d}"
        run_dir = outputs_root / experiment / run_id
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse candidate attempt path: {run_dir}")
        evidence_dir = outputs_root / experiment / "orchestration" / run_id
        evidence_dir.mkdir(parents=True, exist_ok=False)
        seed_plan = evidence_dir / "randomized_seeds.json"
        _atomic_write(seed_plan, json.dumps(_pose_seeds(pose), indent=2) + "\n")
        command = [
            str(LAUNCHER),
            "-p",
            "scripts/run_scripted_baseline.py",
            "--viz",
            "none",
            "--device",
            "cpu",
            "--robot",
            "alex_v2",
            "--episodes",
            "0",
            "--randomized",
            "0",
            "--seed",
            str(pose["source_seed_start"]),
            "--experiment",
            experiment,
            "--run-id",
            run_id,
            "--no-export",
            "--candidate-pool",
            "--randomized-seed-plan",
            str(seed_plan),
            "--door-pose-id",
            pose_id,
            "--door-yaw-deg",
            str(pose["door_yaw_deg"]),
            "--door-offset-x",
            str(pose["door_offset_x_m"]),
            "--door-offset-y",
            str(pose["door_offset_y_m"]),
        ]
        attempt = {
            "attempt": attempt_number,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "seed_plan": str(seed_plan),
            "command": command,
            "status": "RUNNING",
        }
        record["attempts"].append(attempt)
        _atomic_json(_state_path(outputs_root, experiment), state)
        print(f"[generate] {pose_id}: one isolated process, attempt {attempt_number}")
        with (evidence_dir / "stdout.log").open("w") as stdout, (
            evidence_dir / "stderr.log"
        ).open("w") as stderr:
            result = subprocess.run(
                command,
                cwd=paths.REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(paths.REPO_ROOT)},
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        attempt["exit_code"] = result.returncode
        try:
            if result.returncode:
                raise RuntimeError(f"candidate process exited {result.returncode}")
            _verify_candidate_run(run_dir, pose)
        except Exception as error:
            attempt["status"] = "FAILED"
            attempt["error"] = str(error)
            _atomic_json(_state_path(outputs_root, experiment), state)
            raise RuntimeError(
                f"pose {pose_id} candidate generation failed; evidence preserved at "
                f"{evidence_dir}: {error}"
            ) from error
        attempt["status"] = "COMPLETED"
        record["completed"] = str(run_dir)
        _atomic_json(_state_path(outputs_root, experiment), state)


def _select_master(
    plan: dict[str, Any], state: dict[str, Any]
) -> tuple[list[EpisodeBuffer], list[Path], list[dict[str, Any]]]:
    selected: list[EpisodeBuffer] = []
    selected_paths: list[Path] = []
    provenance: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        completed = state["poses"].get(pose_id, {}).get("completed")
        if completed is None:
            raise RuntimeError(f"pose {pose_id} has no completed candidate run")
        run_dir = Path(completed)
        _verify_candidate_run(run_dir, pose)
        episode_paths = sorted((run_dir / "episodes").glob("episode_*.hdf5"))
        candidates = sorted(
            ((read_episode(path), path) for path in episode_paths),
            key=lambda item: item[0].meta.seed,
        )
        primary_stop = pose["source_seed_stop"]
        candidates.sort(key=lambda item: (item[0].meta.seed >= primary_stop, item[0].meta.seed))
        pose_selected: list[tuple[EpisodeBuffer, Path]] = []
        failed_primary: list[int] = []
        for episode, path in candidates:
            seed = episode.meta.seed
            namespace = "source" if seed < primary_stop else "overdraw"
            reasons: list[str] = []
            sanity = check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N)
            reasons.extend(sanity.errors)
            if episode.outcome is None or not episode.outcome.success:
                reasons.append(
                    "task failure: "
                    + (episode.outcome.failure_label if episode.outcome else "missing outcome")
                )
            if episode.extras.get("door_pose_id") != pose_id:
                reasons.append("door pose provenance mismatch")
            try:
                content_group = episode_content_key(load_episode_record(path))
            except Exception as error:
                reasons.append(f"content hash failed: {error}")
                content_group = ""
            if content_group and content_group in seen_groups:
                reasons.append("duplicate trajectory-content group")
            decision = "SKIPPED" if reasons else "ELIGIBLE"
            replacement_for = None
            if not reasons and len(pose_selected) < plan["selected_episodes_per_pose"]:
                decision = "SELECTED"
                pose_selected.append((episode, path))
                seen_groups.add(content_group)
                if namespace == "overdraw" and failed_primary:
                    replacement_for = failed_primary.pop(0)
            elif not reasons:
                decision = "NOT_NEEDED_OVERDRAW"
            elif namespace == "source":
                failed_primary.append(seed)
            provenance.append(
                {
                    "pose_id": pose_id,
                    "seed": seed,
                    "namespace": namespace,
                    "episode_id": episode.meta.episode_id,
                    "source_path": str(path),
                    "content_group_sha256": content_group or None,
                    "decision": decision,
                    "reasons": reasons,
                    "replacement_for_seed": replacement_for,
                }
            )
        required_per_pose = int(plan["selected_episodes_per_pose"])
        if len(pose_selected) != required_per_pose:
            raise RuntimeError(
                f"pose {pose_id} has only {len(pose_selected)} safe successful "
                f"content-distinct candidates; need {required_per_pose}"
            )
        selected.extend(episode for episode, _ in pose_selected)
        selected_paths.extend(path for _, path in pose_selected)
    expected_total = int(plan["selected_episodes_per_pose"]) * len(plan["poses"])
    if len(selected) != expected_total:
        raise RuntimeError(
            f"selected {len(selected)} master episodes, expected {expected_total}"
        )
    return selected, selected_paths, provenance


def _source_fingerprint(paths_: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths_):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def _validate_candidate_provenance(
    plan: dict[str, Any],
    provenance: list[dict[str, Any]],
    *,
    selected_episode_ids: list[str],
    expected_source_fingerprint: str,
    require_source_paths: bool,
) -> dict[str, Any]:
    """Validate the complete candidate ledger and selected-source binding."""
    expected_inventory = {
        (pose["pose_id"], seed): namespace
        for pose in plan["poses"]
        for namespace, start, stop in (
            ("source", pose["source_seed_start"], pose["source_seed_stop"]),
            ("overdraw", pose["overdraw_seed_start"], pose["overdraw_seed_stop"]),
        )
        for seed in range(start, stop)
    }
    if len(provenance) != len(expected_inventory):
        raise ValueError(
            f"candidate provenance inventory has {len(provenance)} rows, "
            f"expected {len(expected_inventory)}"
        )
    required_keys = {
        "pose_id",
        "seed",
        "namespace",
        "episode_id",
        "source_path",
        "content_group_sha256",
        "decision",
        "reasons",
        "replacement_for_seed",
    }
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in provenance:
        if not isinstance(row, dict) or set(row) != required_keys:
            raise ValueError("candidate provenance row keys are invalid")
        key = (str(row["pose_id"]), int(row["seed"]))
        if key in rows:
            raise ValueError(f"candidate provenance inventory duplicates {key}")
        if expected_inventory.get(key) != row["namespace"]:
            raise ValueError(f"candidate provenance namespace/seed inventory mismatch: {key}")
        rows[key] = row
    if set(rows) != set(expected_inventory):
        raise ValueError("candidate provenance inventory is missing or has extra pose/seed rows")

    selected = [row for row in provenance if row["decision"] == "SELECTED"]
    if {str(row["episode_id"]) for row in selected} != set(selected_episode_ids):
        raise ValueError("selected candidate episode IDs differ from the paired exports")
    if len(selected) != len(selected_episode_ids) or len(selected_episode_ids) != len(
        set(selected_episode_ids)
    ):
        raise ValueError("selected candidate episode IDs are duplicated")
    expected_per_pose = int(plan["selected_episodes_per_pose"])
    for pose in plan["poses"]:
        count = sum(row["pose_id"] == pose["pose_id"] for row in selected)
        if count != expected_per_pose:
            raise ValueError(f"pose {pose['pose_id']} selected candidate balance is {count}")

    selected_groups: set[str] = set()
    skipped_sources: dict[tuple[str, int], dict[str, Any]] = {}
    replacement_targets: set[tuple[str, int]] = set()
    selected_paths: list[Path] = []
    allowed = {"SELECTED", "SKIPPED", "NOT_NEEDED_OVERDRAW"}
    for row in provenance:
        decision = row["decision"]
        reasons = row["reasons"]
        if decision not in allowed or not isinstance(reasons, list):
            raise ValueError("candidate provenance decision/reasons are invalid")
        if decision == "SKIPPED" and not reasons:
            raise ValueError("skipped candidate has no deterministic rejection reason")
        if decision != "SKIPPED" and reasons:
            raise ValueError("admitted/not-needed candidate unexpectedly carries rejection reasons")
        if row["namespace"] == "source" and decision == "NOT_NEEDED_OVERDRAW":
            raise ValueError("source candidate cannot be marked not-needed overdraw")
        if decision == "SKIPPED" and row["namespace"] == "source":
            skipped_sources[(row["pose_id"], row["seed"])] = row
        replacement = row["replacement_for_seed"]
        if replacement is not None:
            if decision != "SELECTED" or row["namespace"] != "overdraw":
                raise ValueError("replacement linkage belongs only to selected overdraw")
            target = (row["pose_id"], int(replacement))
            if target not in rows or rows[target]["namespace"] != "source":
                raise ValueError("replacement link does not target a source candidate")
            if rows[target]["decision"] != "SKIPPED" or target in replacement_targets:
                raise ValueError(
                    "replacement link is false, duplicated, or targets admitted source"
                )
            replacement_targets.add(target)
        elif decision == "SELECTED" and row["namespace"] == "overdraw" and skipped_sources:
            raise ValueError("selected overdraw candidate lacks deterministic replacement linkage")

        if decision != "SELECTED":
            continue
        group = row["content_group_sha256"]
        if not isinstance(group, str) or re.fullmatch(r"[0-9a-f]{64}", group) is None:
            raise ValueError("selected candidate content-group hash is invalid")
        if group in selected_groups:
            raise ValueError("selected candidate content-group hashes are duplicated")
        selected_groups.add(group)
        if require_source_paths:
            source_path = Path(str(row["source_path"]))
            if not source_path.is_file():
                raise ValueError(f"selected candidate source path is missing: {source_path}")
            episode = read_episode(source_path)
            if (
                episode.meta.seed != row["seed"]
                or episode.meta.episode_id != row["episode_id"]
                or episode.extras.get("door_pose_id") != row["pose_id"]
            ):
                raise ValueError("selected candidate raw pose/seed/episode provenance mismatch")
            if episode_content_key(load_episode_record(source_path)) != group:
                raise ValueError("selected candidate content-group hash differs from raw source")
            sanity = check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N)
            if sanity.errors or episode.outcome is None or not episode.outcome.success:
                raise ValueError("selected candidate is not safe and successful")
            selected_paths.append(source_path)
    if set(skipped_sources) != replacement_targets:
        raise ValueError("candidate replacement linkage does not cover every skipped source")
    if require_source_paths and _source_fingerprint(selected_paths) != expected_source_fingerprint:
        raise ValueError("selected-source fingerprint differs from the master manifest")
    canonical_ledger = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decisions = {
        name: sum(row["decision"] == name for row in provenance) for name in sorted(allowed)
    }
    return {
        "status": "PASS",
        "candidate_count": len(provenance),
        "selected_count": len(selected),
        "decision_counts": decisions,
        "candidate_provenance_sha256": canonical_ledger,
        "source_fingerprint_sha256": expected_source_fingerprint,
    }


def publish(
    config: SweepConfig,
    plan: dict[str, Any],
    *,
    plan_path: Path,
    outputs_root: Path,
    datasets_root: Path,
    experiment: str,
) -> None:
    source = _git_state()
    if not source["clean_tree"]:
        raise RuntimeError("official scale publication requires a clean committed checkout")
    state_path = _state_path(outputs_root, experiment)
    if not state_path.is_file():
        raise FileNotFoundError(f"generation state is missing: {state_path}")
    state = json.loads(state_path.read_text())
    if state.get("source_git_commit") != source["commit"]:
        raise RuntimeError("generation and publication source commits differ")
    if state.get("pose_plan_sha256") != _plan_hash(plan_path):
        raise RuntimeError("generation state pose plan hash mismatch")
    selected, selected_paths, provenance = _select_master(plan, state)
    source_fp = _source_fingerprint(selected_paths)
    _validate_candidate_provenance(
        plan,
        provenance,
        selected_episode_ids=[episode.meta.episode_id for episode in selected],
        expected_source_fingerprint=source_fp,
        require_source_paths=True,
    )
    robot_asset = dataset_robot_asset_payload(selected)
    per_pose = {
        pose: sum(episode.extras.get("door_pose_id") == pose for episode in selected)
        for pose in config.dataset.pose_ids
    }
    manifest: dict[str, Any] = {
        "schema": MASTER_SCHEMA,
        "task": config.dataset.task,
        "master_version": config.dataset.master_version,
        "obs_preset": config.dataset.obs_preset,
        "source_git": source,
        "pose_plan": str(plan_path),
        "pose_plan_sha256": _plan_hash(plan_path),
        "source_fingerprint_sha256": source_fp,
        "counts": {"total": len(selected), "per_pose": per_pose},
        "robot_asset": robot_asset,
        "candidate_provenance": provenance,
        "selected_episode_ids": sorted(episode.meta.episode_id for episode in selected),
        "action_spaces": {},
        "views": {},
        "normalization_artifacts": {},
        "publication_status": "BUILDING",
    }
    task_root = datasets_root / config.dataset.task
    marker = task_root / "publications" / f"{config.dataset.master_version}.json"
    if not marker.exists():
        export_paired_ee_datasets_atomic(
            selected,
            datasets_root,
            version=config.dataset.master_version,
            manifest=manifest,
        )
    else:
        marker_payload = json.loads(marker.read_text())
        if marker_payload.get("status") not in {"PAIRED_PAYLOADS_ONLY", "COMPLETE"}:
            raise RuntimeError("existing scale publication marker is invalid")

    datasets = {
        space: EpisodeDataset(
            datasets_root / config.dataset.task / space / config.dataset.master_version
        )
        for space in config.dataset.spaces
    }
    reference_ids = datasets["A2_ee_delta"].episode_ids
    if datasets["A3_obj_rel_ee_delta"].episode_ids != reference_ids:
        raise RuntimeError("published A2/A3 source episode IDs differ")
    for space, dataset in datasets.items():
        manifest["action_spaces"][space] = {
            "path": config.dataset.spaces[space],
            "dataset_fingerprint_sha256": dataset_fingerprint(
                dataset, config.dataset.obs_preset
            ),
            "episode_ids": dataset.episode_ids,
        }

    entries = split_entries(datasets["A2_ee_delta"])
    view_counts = {view.view_id: view.train for view in config.views}
    views = build_nested_views(
        entries,
        view_train_counts=view_counts,
        pose_ids=config.dataset.pose_ids,
        seed=config.selection.seed,
        master_version=config.dataset.master_version,
        master_fingerprint=source_fp,
    )
    for view_id, payload in views.items():
        output = view_path(datasets_root, config.dataset.task, view_id)
        if output.exists() and load_view_payload(output) != payload:
            raise RuntimeError(f"existing view payload differs: {output}")
        if not output.exists():
            save_view_payload(output, payload)
        manifest["views"][view_id] = {
            "path": str(output.relative_to(paths.REPO_ROOT)),
            "view_fingerprint_sha256": payload["view_fingerprint_sha256"],
            "counts": payload["counts"],
        }
        for space, dataset in datasets.items():
            norm_path = view_norm_stats_path(dataset.dataset_dir, view_id)
            stats = compute_norm_stats(
                dataset,
                payload["splits"]["train"],
                config.dataset.obs_preset,
                view_id=view_id,
                view_fingerprint=payload["view_fingerprint_sha256"],
            )
            if norm_path.exists():
                loaded = load_norm_stats(norm_path)
                errors = validate_norm_stats(
                    loaded,
                    dataset,
                    payload["splits"]["train"],
                    config.dataset.obs_preset,
                    view_id=view_id,
                    view_fingerprint=payload["view_fingerprint_sha256"],
                )
                if errors:
                    raise RuntimeError(
                        f"existing normalization artifact is stale: {norm_path}: {errors}"
                    )
            else:
                save_norm_stats(norm_path, stats)
            loaded = load_norm_stats(norm_path)
            key = f"{space}:{view_id}"
            manifest["normalization_artifacts"][key] = {
                "path": str(norm_path.relative_to(paths.REPO_ROOT)),
                "sha256": _sha256_file(norm_path),
                "normalization_fingerprint_sha256": loaded.normalization_fingerprint,
                "train_episode_ids": list(loaded.train_episode_ids),
            }

    manifest["publication_status"] = "COMPLETE"
    master_path = task_root / f"{config.dataset.master_version}_manifest.json"
    _atomic_json(master_path, manifest)
    for space in config.dataset.spaces:
        _atomic_json(
            datasets_root
            / config.dataset.task
            / space
            / config.dataset.master_version
            / "manifest.json",
            manifest,
        )
    _atomic_json(
        marker,
        {
            "schema": PUBLICATION_SCHEMA,
            "status": "COMPLETE",
            "task": config.dataset.task,
            "master_version": config.dataset.master_version,
            "source_git_commit": source["commit"],
            "source_fingerprint_sha256": source_fp,
            "master_manifest": str(master_path.relative_to(paths.REPO_ROOT)),
            "master_manifest_sha256": _sha256_file(master_path),
            "view_ids": [view.view_id for view in config.views],
            "normalization_count": 8,
        },
    )
    verify(config, plan, datasets_root=datasets_root)


def verify(config: SweepConfig, plan: dict[str, Any], *, datasets_root: Path) -> dict[str, Any]:
    task_root = datasets_root / config.dataset.task
    marker_path = task_root / "publications" / f"{config.dataset.master_version}.json"
    manifest_path = task_root / f"{config.dataset.master_version}_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("scale master publication marker or manifest is missing")
    marker = json.loads(marker_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if marker.get("schema") != PUBLICATION_SCHEMA or marker.get("status") != "COMPLETE":
        raise RuntimeError("scale publication is not complete")
    if marker.get("master_manifest_sha256") != _sha256_file(manifest_path):
        raise RuntimeError("scale master manifest hash mismatch")
    if manifest.get("schema") != MASTER_SCHEMA or manifest.get("publication_status") != "COMPLETE":
        raise RuntimeError("scale master manifest contract mismatch")
    if manifest.get("pose_plan") != config.selection.pose_plan:
        raise RuntimeError("scale master pose-plan path mismatch")
    if manifest.get("pose_plan_sha256") != _plan_hash(
        paths.REPO_ROOT / config.selection.pose_plan
    ):
        raise RuntimeError("scale master pose-plan hash mismatch")
    if manifest.get("counts") != {
        "total": 550,
        "per_pose": {pose: 110 for pose in config.dataset.pose_ids},
    }:
        raise RuntimeError("scale master count/pose balance mismatch")
    datasets = {
        space: EpisodeDataset(
            datasets_root / config.dataset.task / space / config.dataset.master_version
        )
        for space in config.dataset.spaces
    }
    a2 = datasets["A2_ee_delta"]
    a3 = datasets["A3_obj_rel_ee_delta"]
    if len(a2) != 550 or len(a3) != 550 or a2.episode_ids != a3.episode_ids:
        raise RuntimeError("paired master episode inventory mismatch")
    if not any(
        not np.allclose(a2.by_id(episode_id).actions, a3.by_id(episode_id).actions, atol=1e-12)
        for episode_id in a2.episode_ids
    ):
        raise RuntimeError("A2/A3 paired masters are numerically identical")
    ledger_report = _validate_candidate_provenance(
        plan,
        list(manifest.get("candidate_provenance") or ()),
        selected_episode_ids=list(manifest.get("selected_episode_ids") or ()),
        expected_source_fingerprint=str(manifest.get("source_fingerprint_sha256", "")),
        require_source_paths=True,
    )
    if sorted(manifest["selected_episode_ids"]) != sorted(a2.episode_ids):
        raise RuntimeError("candidate ledger selected IDs differ from paired exports")
    entries = split_entries(a2)
    view_counts = {view.view_id: view.train for view in config.views}
    views = {
        view.view_id: load_view_payload(
            view_path(datasets_root, config.dataset.task, view.view_id)
        )
        for view in config.views
    }
    failures = validate_nested_views(
        views,
        entries,
        view_train_counts=view_counts,
        pose_ids=config.dataset.pose_ids,
        master_version=config.dataset.master_version,
        master_fingerprint=manifest["source_fingerprint_sha256"],
    )
    if failures:
        raise RuntimeError("scale view verification failed: " + "; ".join(failures))
    norm_rows: dict[str, Any] = {}
    for space, dataset in datasets.items():
        expected_dataset_fp = dataset_fingerprint(dataset, config.dataset.obs_preset)
        if manifest["action_spaces"][space]["dataset_fingerprint_sha256"] != expected_dataset_fp:
            raise RuntimeError(f"scale dataset fingerprint mismatch for {space}")
        for view_id, payload in views.items():
            norm_path = view_norm_stats_path(dataset.dataset_dir, view_id)
            stats = load_norm_stats(norm_path)
            errors = validate_norm_stats(
                stats,
                dataset,
                payload["splits"]["train"],
                config.dataset.obs_preset,
                view_id=view_id,
                view_fingerprint=payload["view_fingerprint_sha256"],
            )
            if errors:
                raise RuntimeError(
                    f"normalization validation failed for {space}/{view_id}: {errors}"
                )
            key = f"{space}:{view_id}"
            declared = manifest["normalization_artifacts"][key]
            if declared["sha256"] != _sha256_file(norm_path):
                raise RuntimeError(f"normalization hash mismatch for {key}")
            norm_rows[key] = declared
    report = {
        "schema": "alexdoor_xas.scale_verification.v1",
        "status": "PASS",
        "master_count": 550,
        "per_pose": {pose: 110 for pose in config.dataset.pose_ids},
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
        "action_spaces": manifest["action_spaces"],
        "views": manifest["views"],
        "normalization_artifacts": norm_rows,
        "generation_provenance": {
            **ledger_report,
            "pose_plan": config.selection.pose_plan,
            "pose_plan_sha256": manifest["pose_plan_sha256"],
            "calibration": config.selection.calibration,
            "calibration_fingerprint": plan["calibration_fingerprint"],
        },
    }
    report_path = paths.OUTPUTS_DIR / "cluster_sweep" / "scale_verification.json"
    _atomic_json(report_path, report)
    print(f"PASS: verified 550-episode paired master, four views, and eight norms: {report_path}")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_sweep_config(args.config)
        plan = _load_plan(args.pose_plan, config)
        if args.command == "generate":
            generate(
                config,
                plan,
                plan_path=args.pose_plan,
                outputs_root=args.outputs_root,
                experiment=args.experiment,
            )
        elif args.command == "publish":
            publish(
                config,
                plan,
                plan_path=args.pose_plan,
                outputs_root=args.outputs_root,
                datasets_root=args.datasets_root,
                experiment=args.experiment,
            )
        else:
            verify(config, plan, datasets_root=args.datasets_root)
    except Exception as error:  # noqa: BLE001 - CLI surfaces the fail-closed gate.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
