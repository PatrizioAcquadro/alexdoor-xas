"""Shared scale-generation plan and candidate-provenance evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import (
    RobotAssetRef,
    derive_fixed_base_door_manifest,
    validate_alex_v2_manifest,
)
from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest
from alexdoor_xas.calibration.alex_v2_door import (
    CalibrationError,
    load_validated_alex_v2_door_calibration,
)
from alexdoor_xas.dataset import episode_content_key, load_episode_record
from alexdoor_xas.eval.sanity import FORCE_DATASET_LIMIT_N, check_alex_episode
from alexdoor_xas.recording import EpisodeBuffer, read_episode

from .config import SweepConfig

STATE_SCHEMA = "alexdoor_xas.scale_generation_state.v1"
DEFAULT_EXPERIMENT = "v3_scale_generation"
SCALE_CALIBRATION_RUNTIME_VERSIONS = {
    "isaac_lab": "3.0.0",
    "isaac_sim": "6.0.1-rc.7+release.42383.32955d8d.gl",
}


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
    calibration_path = paths.REPO_ROOT / config.selection.calibration
    master_path = paths.REPO_ROOT / config.dataset.master_manifest
    try:
        if master_path.is_file():
            master = json.loads(master_path.read_text())
            master_asset = master.get("robot_asset")
            if not isinstance(master_asset, dict):
                raise ValueError("scale master robot asset contract is missing")
            runtime_asset = RobotAssetRef.from_dict(master_asset)
        else:
            runtime_manifest = derive_fixed_base_door_manifest(build_alex_v2_manifest())
            runtime_asset = validate_alex_v2_manifest(runtime_manifest)
        calibration = load_validated_alex_v2_door_calibration(
            calibration_path,
            runtime_asset=runtime_asset,
            runtime_versions=SCALE_CALIBRATION_RUNTIME_VERSIONS,
        )
    except (CalibrationError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"scale calibration validation failed: {error}") from error
    if plan["calibration_fingerprint"] != calibration.payload["fingerprint"]:
        raise ValueError("scale calibration fingerprint differs from the canonical artifact")
    canonical = json.loads((paths.REPO_ROOT / config.selection.canonical_pose_plan).read_text())
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


def _state_path(outputs_root: Path, experiment: str) -> Path:
    return outputs_root / experiment / "generation_state.json"


def _validate_generation_state_binding(
    state: dict[str, Any],
    *,
    pose_plan_sha256: str,
    source_git_commit: str,
) -> None:
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("scale generation state schema mismatch")
    if state.get("pose_plan_sha256") != pose_plan_sha256:
        raise ValueError("scale generation state pose-plan hash mismatch")
    if state.get("source_git_commit") != source_git_commit:
        raise ValueError("scale generation state source commit mismatch")


def _pose_seeds(pose: dict[str, Any]) -> list[int]:
    return [
        *range(pose["source_seed_start"], pose["source_seed_stop"]),
        *range(pose["overdraw_seed_start"], pose["overdraw_seed_stop"]),
    ]


def _validate_generation_evidence_path(
    path: Path,
    generation_root: Path,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    try:
        resolved_root = generation_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"scale generation root is unavailable: {generation_root}") from error
    if not resolved_root.is_dir():
        raise ValueError(f"scale generation root is not a directory: {generation_root}")
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside the scale generation root: {path}") from error
    if directory:
        if not path.is_dir():
            raise ValueError(f"{label} is not a directory: {path}")
    elif not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return resolved


def _verify_candidate_run(
    run_dir: Path,
    pose: dict[str, Any],
    *,
    generation_root: Path,
) -> list[Path]:
    run_dir = _validate_generation_evidence_path(
        run_dir,
        generation_root,
        label=f"pose {pose['pose_id']} candidate run directory",
        directory=True,
    )
    files = sorted((run_dir / "episodes").glob("episode_*.hdf5"))
    expected = _pose_seeds(pose)
    if len(files) != len(expected):
        raise ValueError(
            f"pose {pose['pose_id']} candidate run has {len(files)} episodes, "
            f"expected {len(expected)}"
        )
    validated_files: list[Path] = []
    for path in files:
        validated_files.append(
            _validate_generation_evidence_path(
                path,
                generation_root,
                label=f"pose {pose['pose_id']} candidate HDF5 evidence",
            )
        )
        _validate_generation_evidence_path(
            path.with_suffix(".meta.json"),
            generation_root,
            label=f"pose {pose['pose_id']} candidate metadata evidence",
        )
    episodes = [read_episode(path) for path in validated_files]
    seeds = sorted(episode.meta.seed for episode in episodes)
    if seeds != sorted(expected):
        raise ValueError(f"pose {pose['pose_id']} candidate seed inventory mismatch")
    if any(episode.extras.get("door_pose_id") != pose["pose_id"] for episode in episodes):
        raise ValueError(f"pose {pose['pose_id']} candidate records wrong pose provenance")
    sanity_path = _validate_generation_evidence_path(
        run_dir / "metrics" / "sanity.json",
        generation_root,
        label=f"pose {pose['pose_id']} candidate sanity evidence",
    )
    sanity = json.loads(sanity_path.read_text())
    if sanity.get("n_episodes_checked") != len(expected):
        raise ValueError(f"pose {pose['pose_id']} sanity candidate count mismatch")
    return validated_files


def _candidate_paths_from_state(
    plan: dict[str, Any],
    state: dict[str, Any],
    *,
    generation_root: Path,
) -> dict[str, list[Path]]:
    paths_by_pose: dict[str, list[Path]] = {}
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        completed = state.get("poses", {}).get(pose_id, {}).get("completed")
        if completed is None:
            raise RuntimeError(f"pose {pose_id} has no completed candidate run")
        paths_by_pose[pose_id] = _verify_candidate_run(
            Path(completed),
            pose,
            generation_root=generation_root,
        )
    return paths_by_pose


def _evaluate_candidate_paths(
    plan: dict[str, Any], paths_by_pose: dict[str, list[Path]]
) -> tuple[list[EpisodeBuffer], list[Path], list[dict[str, Any]]]:
    """Apply the frozen selection algorithm and emit its canonical complete ledger."""
    selected: list[EpisodeBuffer] = []
    selected_paths: list[Path] = []
    provenance: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        candidates = sorted(
            ((read_episode(path), path) for path in paths_by_pose[pose_id]),
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
        raise RuntimeError(f"selected {len(selected)} master episodes, expected {expected_total}")
    return selected, selected_paths, provenance


def _select_master(
    plan: dict[str, Any],
    state: dict[str, Any],
    *,
    generation_root: Path,
) -> tuple[list[EpisodeBuffer], list[Path], list[dict[str, Any]]]:
    return _evaluate_candidate_paths(
        plan,
        _candidate_paths_from_state(plan, state, generation_root=generation_root),
    )


def _source_fingerprint(paths_: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths_):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def _raw_candidate_evidence_fingerprint(
    plan: dict[str, Any], paths_by_pose: dict[str, list[Path]]
) -> str:
    rows: list[dict[str, Any]] = []
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        for path in paths_by_pose[pose_id]:
            episode = read_episode(path)
            rows.append(
                {
                    "pose_id": pose_id,
                    "seed": episode.meta.seed,
                    "hdf5_sha256": _sha256_file(path),
                }
            )
    rows.sort(key=lambda row: (row["pose_id"], row["seed"]))
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_candidate_provenance(
    plan: dict[str, Any],
    provenance: list[dict[str, Any]],
    *,
    selected_episode_ids: list[str],
    expected_source_fingerprint: str,
    require_source_paths: bool,
    candidate_state: dict[str, Any] | None = None,
    generation_root: Path | None = None,
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
    row_order: list[tuple[str, int]] = []
    for row in provenance:
        if not isinstance(row, dict) or set(row) != required_keys:
            raise ValueError("candidate provenance row keys are invalid")
        key = (str(row["pose_id"]), int(row["seed"]))
        if key in rows:
            raise ValueError(f"candidate provenance inventory duplicates {key}")
        if expected_inventory.get(key) != row["namespace"]:
            raise ValueError(f"candidate provenance namespace/seed inventory mismatch: {key}")
        rows[key] = row
        row_order.append(key)
    if set(rows) != set(expected_inventory):
        raise ValueError("candidate provenance inventory is missing or has extra pose/seed rows")
    if row_order != list(expected_inventory):
        raise ValueError("candidate provenance row order differs from the deterministic inventory")

    raw_replay: dict[str, Any] | None = None
    replay_selected_paths: list[Path] = []
    if require_source_paths:
        if candidate_state is None:
            raise ValueError("candidate raw replay requires the completed generation state")
        if generation_root is None:
            raise ValueError("candidate raw replay requires the authoritative generation root")
        paths_by_pose = _candidate_paths_from_state(
            plan,
            candidate_state,
            generation_root=generation_root,
        )
        replay_selected, replay_selected_paths, replay_provenance = _evaluate_candidate_paths(
            plan, paths_by_pose
        )
        if provenance != replay_provenance:
            for index, (declared, expected) in enumerate(
                zip(provenance, replay_provenance, strict=False)
            ):
                if declared != expected:
                    fields = sorted(
                        key
                        for key in set(declared) | set(expected)
                        if declared.get(key) != expected.get(key)
                    )
                    field_labels = [field.replace("_", " ") for field in fields]
                    raise ValueError(
                        "candidate raw replay mismatch at row "
                        f"{index} ({expected.get('pose_id')}, {expected.get('seed')}): "
                        f"fields={field_labels}"
                    )
            raise ValueError("candidate raw replay row count mismatch")
        replay_ids = [episode.meta.episode_id for episode in replay_selected]
        if sorted(replay_ids) != sorted(selected_episode_ids):
            raise ValueError("candidate raw replay selected episode inventory mismatch")
        raw_replay = {
            "status": "PASS",
            "candidate_count": len(replay_provenance),
            "candidate_evidence_sha256": _raw_candidate_evidence_fingerprint(
                plan, paths_by_pose
            ),
        }
        if candidate_state.get("pose_plan_sha256") is not None:
            raw_replay["pose_plan_sha256"] = candidate_state["pose_plan_sha256"]
        if candidate_state.get("source_git_commit") is not None:
            raw_replay["source_git_commit"] = candidate_state["source_git_commit"]

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
    if set(skipped_sources) != replacement_targets:
        raise ValueError("candidate replacement linkage does not cover every skipped source")
    if (
        require_source_paths
        and _source_fingerprint(replay_selected_paths) != expected_source_fingerprint
    ):
        raise ValueError("selected-source fingerprint differs from the master manifest")
    canonical_ledger = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decisions = {
        name: sum(row["decision"] == name for row in provenance) for name in sorted(allowed)
    }
    report = {
        "status": "PASS",
        "candidate_count": len(provenance),
        "selected_count": len(selected),
        "decision_counts": decisions,
        "candidate_provenance_sha256": canonical_ledger,
        "source_fingerprint_sha256": expected_source_fingerprint,
    }
    if raw_replay is not None:
        report["raw_replay"] = raw_replay
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_EXPERIMENT",
    "SCALE_CALIBRATION_RUNTIME_VERSIONS",
    "STATE_SCHEMA",
    "_candidate_paths_from_state",
    "_evaluate_candidate_paths",
    "_load_plan",
    "_pose_seeds",
    "_select_master",
    "_source_fingerprint",
    "_state_path",
    "_validate_candidate_provenance",
    "_validate_generation_state_binding",
    "_verify_candidate_run",
]
