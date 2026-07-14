"""Exact clean-tree transfer inventory for the nested dataset-scale sweep."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.transfer import (
    git_state,
    sha256_file,
)
from alexdoor_xas.cluster_pilot.transfer import (
    secret_problems as _pilot_secret_problems,
)
from alexdoor_xas.dataset import (
    EpisodeDataset,
    dataset_fingerprint,
    load_view_payload,
    split_entries,
    validate_nested_views,
    view_norm_stats_path,
    view_path,
)
from alexdoor_xas.policies.common.data import PolicyDataError, load_policy_data

from .config import SweepConfig

TRANSFER_SCHEMA = "alexdoor_xas.cluster_sweep_transfer_manifest.v1"
DEFAULT_OUTPUT_DIR = Path("outputs/cluster_sweep")
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "sweep_transfer_manifest.json"
DEFAULT_FILE_LIST_PATH = DEFAULT_OUTPUT_DIR / "rsync-files.txt"
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


class SweepTransferError(ValueError):
    """Raised when the sweep package cannot be proven exact and safe."""


def secret_problems(relative: str, content: bytes) -> list[str]:
    """Apply the pilot scanner while excluding four known ML-token assignments.

    The pilot's byte regex intentionally treats token-named assignments as
    credential-like. ACT and Diffusion use that conventional name for tensor
    embeddings. Only these exact code-shaped assignments are removed before
    scanning; all other source and every non-Python payload remain unchanged.
    """
    if Path(relative).suffix != ".py":
        return _pilot_secret_problems(relative, content)
    safe_names = (b"cls_token", b"obs_token", b"z_token", b"t_token")
    sanitized: list[bytes] = []
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        safe_assignment = any(
            stripped.startswith(name + b" = self.")
            or stripped.startswith(b"self." + name + b" = nn.")
            for name in safe_names
        )
        sanitized.append(b"\n" if safe_assignment else line)
    return _pilot_secret_problems(relative, b"".join(sanitized))


def build_sweep_transfer_manifest(
    repo_root: str | Path,
    config: SweepConfig,
    *,
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    state = dict(source_state) if source_state is not None else git_state(root)
    _validate_source_state(state)
    files, dataset_contract, robot_asset = _collect_contract(
        root,
        config,
        require_tracked=source_state is None,
        require_local_sources=True,
    )
    entries = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "category": category,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, category in files
        ),
        key=lambda entry: entry["path"],
    )
    manifest = {
        "schema": TRANSFER_SCHEMA,
        "sweep_config_schema": config.schema,
        "sweep_id": config.sweep_id,
        "created_utc": state["commit_time"],
        "source_git": state,
        "dataset": dataset_contract,
        "robot_asset_provenance": robot_asset,
        "files": entries,
        "category_counts": dict(
            sorted(Counter(entry["category"] for entry in entries).items())
        ),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "verification": {"algorithm": "sha256", "status": "PASS"},
    }
    failures = verify_sweep_transfer_manifest(
        manifest,
        root,
        config,
        source_state=state,
        require_tracked=source_state is None,
    )
    if failures:
        raise SweepTransferError("sweep transfer self-verification failed: " + "; ".join(failures))
    return manifest


def verify_sweep_transfer_manifest(
    manifest: dict[str, Any],
    repo_root: str | Path,
    config: SweepConfig,
    *,
    source_state: dict[str, Any] | None = None,
    require_tracked: bool = False,
) -> list[str]:
    root = Path(repo_root).resolve()
    state = dict(source_state) if source_state is not None else git_state(root)
    failures: list[str] = []
    try:
        _validate_source_state(state)
    except SweepTransferError as error:
        failures.append(str(error))
    if manifest.get("schema") != TRANSFER_SCHEMA:
        failures.append("sweep transfer schema mismatch")
    if manifest.get("sweep_config_schema") != config.schema:
        failures.append("sweep config schema mismatch")
    if manifest.get("sweep_id") != config.sweep_id:
        failures.append("sweep id mismatch")
    declared_source = manifest.get("source_git") or {}
    if declared_source.get("commit") != state.get("commit"):
        failures.append("source commit mismatch")
    if declared_source.get("clean_tree") is not True:
        failures.append("manifest source must be clean")
    if manifest.get("created_utc") != declared_source.get("commit_time"):
        failures.append("manifest timestamp must equal the source commit timestamp")
    try:
        expected_files, expected_dataset, expected_asset = _collect_contract(
            root,
            config,
            require_tracked=require_tracked,
            require_local_sources=False,
        )
    except (OSError, ValueError, KeyError, PolicyDataError, SweepTransferError) as error:
        failures.append(f"cannot reconstruct sweep contract: {error}")
        return failures
    if manifest.get("dataset") != expected_dataset:
        failures.append("dataset/view/normalization contract mismatch")
    if manifest.get("robot_asset_provenance") != expected_asset:
        failures.append("Alex V2 URDF provenance mismatch")
    expected = {path.relative_to(root).as_posix(): category for path, category in expected_files}
    declared_entries = manifest.get("files")
    if not isinstance(declared_entries, list):
        failures.append("manifest files must be a list")
        declared_entries = []
    actual: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, entry in enumerate(declared_entries):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is not an object")
            continue
        relative = entry.get("path")
        category = entry.get("category")
        if not isinstance(relative, str) or not isinstance(category, str):
            failures.append(f"files[{index}] path/category is invalid")
            continue
        if relative in actual:
            failures.append(f"duplicate manifest path: {relative}")
            continue
        actual[relative] = category
        counts[category] += 1
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"manifest path escapes repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing manifest payload: {relative}")
            continue
        if entry.get("size_bytes") != path.stat().st_size:
            failures.append(f"size mismatch: {relative}")
        if entry.get("sha256") != sha256_file(path):
            failures.append(f"hash mismatch: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            failures.extend(secret_problems(relative, path.read_bytes()))
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        failures.append(f"exact inventory mismatch: missing={missing}, extra={extra}")
    for relative in set(actual) & set(expected):
        if actual[relative] != expected[relative]:
            failures.append(f"category mismatch: {relative}")
    if manifest.get("category_counts") != dict(sorted(counts.items())):
        failures.append("category_counts mismatch")
    if manifest.get("file_count") != len(declared_entries):
        failures.append("file_count mismatch")
    if manifest.get("total_size_bytes") != sum(
        entry.get("size_bytes", 0) for entry in declared_entries if isinstance(entry, dict)
    ):
        failures.append("total_size_bytes mismatch")
    if manifest.get("verification") != {"algorithm": "sha256", "status": "PASS"}:
        failures.append("verification declaration mismatch")
    return failures


def _collect_contract(
    root: Path,
    config: SweepConfig,
    *,
    require_tracked: bool,
    require_local_sources: bool,
) -> tuple[list[tuple[Path, str]], dict[str, Any], dict[str, Any]]:
    inventory: list[tuple[Path, str]] = []
    task_root = root / "datasets" / config.dataset.task
    marker_path = task_root / "publications" / f"{config.dataset.master_version}.json"
    master_path = root / config.dataset.master_manifest
    for path in (marker_path, master_path):
        if not path.is_file():
            raise SweepTransferError(f"missing official scale metadata: {path}")
        inventory.append((path, "master_metadata"))
    marker = json.loads(marker_path.read_text())
    master = json.loads(master_path.read_text())
    if marker.get("status") != "COMPLETE" or master.get("publication_status") != "COMPLETE":
        raise SweepTransferError("scale master is not atomically complete")
    if marker.get("master_manifest_sha256") != sha256_file(master_path):
        raise SweepTransferError("scale master manifest hash is stale")
    if master.get("counts") != {
        "total": 550,
        "per_pose": {pose: 110 for pose in config.dataset.pose_ids},
    }:
        raise SweepTransferError("scale master count contract drifted")
    pose_plan_path = root / config.selection.pose_plan
    calibration_path = root / config.selection.calibration
    canonical_pose_path = root / config.selection.canonical_pose_plan
    for path in (pose_plan_path, calibration_path, canonical_pose_path):
        if not path.is_file():
            raise SweepTransferError(f"generation contract source is missing: {path}")
    if master.get("pose_plan") != config.selection.pose_plan:
        raise SweepTransferError("scale master pose-plan path is stale")
    if master.get("pose_plan_sha256") != sha256_file(pose_plan_path):
        raise SweepTransferError("scale master pose-plan hash is stale")
    from scripts.build_scale_dataset import (
        DEFAULT_EXPERIMENT,
        _load_plan,
        _state_path,
        _validate_candidate_provenance,
        _validate_generation_state_binding,
    )

    plan = _load_plan(pose_plan_path, config)
    source_ids = list(master.get("selected_episode_ids") or ())
    if len(source_ids) != 550 or len(source_ids) != len(set(source_ids)):
        raise SweepTransferError("scale master selected episode inventory is invalid")
    candidate_state = None
    generation_root = None
    if require_local_sources:
        generation_root = root / "outputs" / DEFAULT_EXPERIMENT
        state_path = _state_path(root / "outputs", DEFAULT_EXPERIMENT)
        if not state_path.is_file():
            raise SweepTransferError(f"scale generation state is missing: {state_path}")
        candidate_state = json.loads(state_path.read_text())
        try:
            _validate_generation_state_binding(
                candidate_state,
                pose_plan_sha256=master["pose_plan_sha256"],
                source_git_commit=master["source_git"]["commit"],
            )
        except ValueError as error:
            raise SweepTransferError(str(error)) from error
    try:
        ledger_report = _validate_candidate_provenance(
            plan,
            list(master.get("candidate_provenance") or ()),
            selected_episode_ids=source_ids,
            expected_source_fingerprint=master["source_fingerprint_sha256"],
            require_source_paths=require_local_sources,
            candidate_state=candidate_state,
            generation_root=generation_root,
        )
    except ValueError as error:
        raise SweepTransferError(f"candidate provenance ledger failed: {error}") from error
    verification_path = root / DEFAULT_OUTPUT_DIR / "scale_verification.json"
    if not verification_path.is_file():
        raise SweepTransferError("local scale generation verification report is missing")
    verification = json.loads(verification_path.read_text())
    expected_generation = {
        **ledger_report,
        "pose_plan": config.selection.pose_plan,
        "pose_plan_sha256": master["pose_plan_sha256"],
        "calibration": config.selection.calibration,
        "calibration_fingerprint": plan["calibration_fingerprint"],
    }
    actual_generation = verification.get("generation_provenance")
    if not isinstance(actual_generation, dict):
        raise SweepTransferError("scale generation verification report is malformed")
    for key, value in expected_generation.items():
        if actual_generation.get(key) != value:
            raise SweepTransferError("scale generation verification report is stale")
    raw_replay = actual_generation.get("raw_replay")
    if (
        not isinstance(raw_replay, dict)
        or raw_replay.get("status") != "PASS"
        or raw_replay.get("candidate_count") != ledger_report["candidate_count"]
        or not isinstance(raw_replay.get("candidate_evidence_sha256"), str)
        or len(raw_replay["candidate_evidence_sha256"]) != 64
        or raw_replay.get("pose_plan_sha256") != master["pose_plan_sha256"]
        or raw_replay.get("source_git_commit") != master["source_git"]["commit"]
    ):
        raise SweepTransferError("scale generation raw replay evidence is stale")
    inventory.append((verification_path, "generation_verification"))

    datasets: dict[str, EpisodeDataset] = {}
    spaces_contract: dict[str, Any] = {}
    for space, relative in config.dataset.spaces.items():
        directory = root / relative
        dataset = EpisodeDataset(directory)
        datasets[space] = dataset
        if sorted(dataset.episode_ids) != sorted(source_ids):
            raise SweepTransferError(f"{space} episode IDs differ from the master manifest")
        expected_paths = {directory / "meta.json", directory / "manifest.json"}
        for episode_id in source_ids:
            prefix = episode_id.split("-", 1)[0]
            expected_paths.update(
                {directory / f"episode_{prefix}.hdf5", directory / f"episode_{prefix}.meta.json"}
            )
        view_files: list[Path] = []
        for view in config.views:
            norm = view_norm_stats_path(directory, view.view_id)
            expected_paths.add(norm)
            view_files.append(norm)
        actual_paths = {path for path in directory.rglob("*") if path.is_file()}
        if actual_paths != expected_paths:
            raise SweepTransferError(
                f"exact dataset inventory mismatch for {space}: "
                f"missing={sorted(str(p) for p in expected_paths - actual_paths)}, "
                f"extra={sorted(str(p) for p in actual_paths - expected_paths)}"
            )
        for path in sorted(expected_paths):
            category = "normalization" if path in view_files else (
                "dataset_metadata"
                if path.name in {"meta.json", "manifest.json"}
                else "dataset_episode"
            )
            inventory.append((path, category))
        declared = master["action_spaces"][space]
        live_action_fingerprint = dataset_fingerprint(dataset, config.dataset.obs_preset)
        if declared.get("dataset_fingerprint_sha256") != live_action_fingerprint:
            raise SweepTransferError(f"master action dataset fingerprint is stale: {space}")
        spaces_contract[space] = {
            "path": relative,
            "source_fingerprint_sha256": master["source_fingerprint_sha256"],
            "dataset_fingerprint_sha256": live_action_fingerprint,
            "episode_ids": sorted(source_ids),
        }
    a2, a3 = datasets["A2_ee_delta"], datasets["A3_obj_rel_ee_delta"]
    if not any(
        not np.allclose(a2.by_id(episode_id).actions, a3.by_id(episode_id).actions, atol=1e-12)
        for episode_id in source_ids
    ):
        raise SweepTransferError("paired A2/A3 master actions are numerically identical")

    views_payload: dict[str, dict[str, Any]] = {}
    for view in config.views:
        path = view_path(root / "datasets", config.dataset.task, view.view_id)
        payload = load_view_payload(path)
        views_payload[view.view_id] = payload
        inventory.append((path, "dataset_view"))
    failures = validate_nested_views(
        views_payload,
        split_entries(a2),
        view_train_counts={view.view_id: view.train for view in config.views},
        pose_ids=config.dataset.pose_ids,
        master_version=config.dataset.master_version,
        master_fingerprint=master["source_fingerprint_sha256"],
    )
    if failures:
        raise SweepTransferError("scale nested-view contract failed: " + "; ".join(failures))

    norms: dict[str, Any] = {}
    for space in config.dataset.spaces:
        for view in config.views:
            cfg = SimpleNamespace(
                task=config.dataset.task,
                space=space,
                version=config.dataset.master_version,
                view_id=view.view_id,
                obs_preset=config.dataset.obs_preset,
            )
            data = load_policy_data(cfg, datasets_root=root / "datasets")
            norm_path = view_norm_stats_path(data.dataset.dataset_dir, view.view_id)
            key = f"{space}:{view.view_id}"
            norms[key] = {
                "path": norm_path.relative_to(root).as_posix(),
                "sha256": sha256_file(norm_path),
                "normalization_fingerprint_sha256": data.stats.normalization_fingerprint,
                "train_episode_ids": list(data.train_ids),
            }
            if norms[key] != master["normalization_artifacts"][key]:
                raise SweepTransferError(f"master normalization contract is stale: {key}")

    for relative in config.tracked_transfer_files:
        path = root / relative
        if not path.is_file():
            raise SweepTransferError(f"tracked sweep source is missing: {relative}")
        if require_tracked:
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise SweepTransferError(f"sweep source is not tracked: {relative}")
        inventory.append((path, "sweep_source"))

    urdf = paths.ALEX_V2_URDF
    if not urdf.is_file():
        raise SweepTransferError(f"Alex V2 URDF is missing: {urdf}")
    asset = master.get("robot_asset") or {}
    robot_asset = _robot_asset_contract(asset, urdf)
    dataset_contract = {
        "task": config.dataset.task,
        "master_version": config.dataset.master_version,
        "obs_preset": config.dataset.obs_preset,
        "counts": master["counts"],
        "source_fingerprint_sha256": master["source_fingerprint_sha256"],
        "spaces": spaces_contract,
        "views": {
            view_id: {
                "path": view_path(root / "datasets", config.dataset.task, view_id)
                .relative_to(root)
                .as_posix(),
                "view_fingerprint_sha256": payload["view_fingerprint_sha256"],
                "counts": payload["counts"],
                "splits": payload["splits"],
            }
            for view_id, payload in views_payload.items()
        },
        "normalization_artifacts": norms,
        "cell_mapping": [
            {
                "index": cell.index,
                "policy": cell.policy,
                "space": cell.space,
                "view_id": cell.view_id,
                "run_id": cell.run_id,
            }
            for cell in config.cells
        ],
        "generation_provenance": {
            **actual_generation,
            "verification_report": verification_path.relative_to(root).as_posix(),
            "verification_report_sha256": sha256_file(verification_path),
            "calibration_sha256": sha256_file(calibration_path),
            "canonical_pose_plan": config.selection.canonical_pose_plan,
            "canonical_pose_plan_sha256": sha256_file(canonical_pose_path),
        },
    }
    if len({path.resolve() for path, _ in inventory}) != len(inventory):
        raise SweepTransferError("transfer inventory contains duplicate files")
    return inventory, dataset_contract, robot_asset


def _robot_asset_contract(asset: dict[str, Any], urdf: Path) -> dict[str, Any]:
    """Bind both the runtime-variant fingerprint and the underlying URDF bytes."""
    manifest = asset.get("manifest")
    if not isinstance(manifest, dict):
        raise SweepTransferError("scale master robot asset manifest is missing")
    runtime_fingerprint = asset.get("sha256")
    if (
        not isinstance(runtime_fingerprint, str)
        or manifest.get("fingerprint") != runtime_fingerprint
        or asset.get("manifest_fingerprint") != runtime_fingerprint
    ):
        raise SweepTransferError("scale master runtime asset fingerprint is inconsistent")
    urdf_sha256 = manifest.get("urdf_sha256")
    if not isinstance(urdf_sha256, str) or len(urdf_sha256) != 64:
        raise SweepTransferError("scale master URDF provenance is missing")
    if sha256_file(urdf) != urdf_sha256:
        raise SweepTransferError("live Alex V2 URDF hash differs from master provenance")
    return {
        "runtime_asset_id": asset.get("id"),
        "runtime_asset_fingerprint_sha256": runtime_fingerprint,
        "urdf_sha256": urdf_sha256,
        "required_cluster_path": "/home/pacquadr/Desktop/Alex/urdf/alex_v2.urdf",
        "transferred": False,
    }


def sweep_rsync_file_list(manifest: dict[str, Any]) -> list[str]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise SweepTransferError("manifest files are missing")
    paths_ = sorted(entry["path"] for entry in entries)
    if len(paths_) != len(set(paths_)):
        raise SweepTransferError("manifest file list contains duplicates")
    return [*paths_, DEFAULT_MANIFEST_PATH.as_posix(), DEFAULT_FILE_LIST_PATH.as_posix()]


def sweep_rsync_template() -> str:
    return (
        "rsync -avP --partial --checksum "
        f"--files-from={DEFAULT_FILE_LIST_PATH.as_posix()} "
        "./ <user>@<host>:<remote_root>/"
    )


def write_sweep_transfer_artifacts(
    repo_root: str | Path,
    manifest: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    root = Path(repo_root).resolve()
    directory = (root / output_dir).resolve()
    directory.relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / DEFAULT_MANIFEST_PATH.name
    file_list = directory / DEFAULT_FILE_LIST_PATH.name
    command = directory / "rsync-command.txt"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(file_list, "\n".join(sweep_rsync_file_list(manifest)) + "\n")
    _atomic_write(command, sweep_rsync_template() + "\n")
    return manifest_path, file_list, command


def _validate_source_state(state: dict[str, Any]) -> None:
    if state.get("clean_tree") is not True:
        raise SweepTransferError("sweep transfer requires a clean source tree")
    commit = state.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise SweepTransferError("sweep transfer source commit is invalid")
    if not state.get("commit_time"):
        raise SweepTransferError("sweep transfer source commit timestamp is missing")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_FILE_LIST_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "TRANSFER_SCHEMA",
    "SweepTransferError",
    "build_sweep_transfer_manifest",
    "secret_problems",
    "sha256_file",
    "sweep_rsync_file_list",
    "sweep_rsync_template",
    "verify_sweep_transfer_manifest",
    "write_sweep_transfer_artifacts",
]
