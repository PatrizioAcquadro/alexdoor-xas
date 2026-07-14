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
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.cluster_sweep import scale_provenance as _scale_provenance
from alexdoor_xas.cluster_sweep.config import SweepConfig, load_sweep_config
from alexdoor_xas.cluster_sweep.scale_provenance import (
    DEFAULT_EXPERIMENT,
    STATE_SCHEMA,
    _load_plan,
    _pose_seeds,
    _select_master,
    _source_fingerprint,
    _state_path,
    _validate_candidate_provenance,
    _validate_generation_state_binding,
    _verify_candidate_run,
)
from alexdoor_xas.data_engine import export_paired_ee_datasets_atomic
from alexdoor_xas.dataset import (
    EpisodeDataset,
    build_nested_views,
    compute_norm_stats,
    dataset_fingerprint,
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

_candidate_paths_from_state = _scale_provenance._candidate_paths_from_state
read_episode = _scale_provenance.read_episode

MASTER_SCHEMA = "alexdoor_xas.scale_master_manifest.v1"
PUBLICATION_SCHEMA = "alexdoor_xas.scale_publication.v1"
DEFAULT_CONFIG = Path("configs/cluster_sweep.v1.json")
DEFAULT_PLAN = Path("configs/door_pose_plan_v3_scale.json")
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
    generation_root = outputs_root / experiment
    state = _load_or_create_state(outputs_root, experiment, plan_path, source)
    for pose in plan["poses"]:
        pose_id = pose["pose_id"]
        record = state["poses"].setdefault(pose_id, {"attempts": [], "completed": None})
        if record["completed"] is not None:
            completed = Path(record["completed"])
            _verify_candidate_run(completed, pose, generation_root=generation_root)
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
            _verify_candidate_run(run_dir, pose, generation_root=generation_root)
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
    _validate_generation_state_binding(
        state,
        pose_plan_sha256=_plan_hash(plan_path),
        source_git_commit=source["commit"],
    )
    if state.get("source_git_commit") != source["commit"]:
        raise RuntimeError("generation and publication source commits differ")
    if state.get("pose_plan_sha256") != _plan_hash(plan_path):
        raise RuntimeError("generation state pose plan hash mismatch")
    generation_root = outputs_root / experiment
    selected, selected_paths, provenance = _select_master(
        plan,
        state,
        generation_root=generation_root,
    )
    source_fp = _source_fingerprint(selected_paths)
    _validate_candidate_provenance(
        plan,
        provenance,
        selected_episode_ids=[episode.meta.episode_id for episode in selected],
        expected_source_fingerprint=source_fp,
        require_source_paths=True,
        candidate_state=state,
        generation_root=generation_root,
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
    generation_root = paths.OUTPUTS_DIR / DEFAULT_EXPERIMENT
    state_path = _state_path(paths.OUTPUTS_DIR, DEFAULT_EXPERIMENT)
    if not state_path.is_file():
        raise RuntimeError(f"scale generation state is missing: {state_path}")
    candidate_state = json.loads(state_path.read_text())
    _validate_generation_state_binding(
        candidate_state,
        pose_plan_sha256=manifest["pose_plan_sha256"],
        source_git_commit=manifest["source_git"]["commit"],
    )
    ledger_report = _validate_candidate_provenance(
        plan,
        list(manifest.get("candidate_provenance") or ()),
        selected_episode_ids=list(manifest.get("selected_episode_ids") or ()),
        expected_source_fingerprint=str(manifest.get("source_fingerprint_sha256", "")),
        require_source_paths=True,
        candidate_state=candidate_state,
        generation_root=generation_root,
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
