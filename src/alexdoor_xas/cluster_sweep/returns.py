"""Exact-attempt 16-cell return manifest and checkpoint verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alexdoor_xas.cluster_pilot.transfer import secret_problems, sha256_file

from .config import SweepCell, SweepConfig

RETURN_SCHEMA = "alexdoor_xas.cluster_sweep_return_manifest.v1"
STATUS_SCHEMA = "alexdoor_xas.cluster_sweep_cell_status.v1"
ATTEMPT_RE = re.compile(r"^[0-9]+$")
ALLOWED_TOP_LEVEL = {
    "checkpoints",
    "environment",
    "logs",
    "metrics",
    "plots",
    "slurm",
    "status",
    "wandb",
}
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".lock", ".log"}


class SweepReturnError(ValueError):
    """Raised when a selected sweep attempt is incomplete or inconsistent."""


def build_sweep_return_manifest(
    results_root: str | Path,
    *,
    attempt_id: str,
    config: SweepConfig,
    transfer_manifest: dict[str, Any],
) -> dict[str, Any]:
    root = Path(results_root).resolve()
    attempt = _attempt_id(attempt_id)
    source_commit = _transfer_source_commit(transfer_manifest, config)
    files, cells = _collect_attempt(root, attempt, config, source_commit)
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
        "schema": RETURN_SCHEMA,
        "sweep_id": config.sweep_id,
        "attempt_id": attempt,
        "source_git_commit": source_commit,
        "cell_count": 16,
        "cells": cells,
        "files": entries,
        "category_counts": dict(
            sorted(Counter(entry["category"] for entry in entries).items())
        ),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "verification": {"algorithm": "sha256", "status": "PASS"},
    }
    failures = verify_sweep_return_manifest(
        manifest,
        root,
        attempt_id=attempt,
        config=config,
        transfer_manifest=transfer_manifest,
    )
    if failures:
        raise SweepReturnError("sweep return self-verification failed: " + "; ".join(failures))
    return manifest


def verify_sweep_return_manifest(
    manifest: dict[str, Any],
    results_root: str | Path,
    *,
    attempt_id: str,
    config: SweepConfig,
    transfer_manifest: dict[str, Any],
) -> list[str]:
    root = Path(results_root).resolve()
    attempt = _attempt_id(attempt_id)
    failures: list[str] = []
    source_commit = _transfer_source_commit(transfer_manifest, config)
    if manifest.get("schema") != RETURN_SCHEMA:
        failures.append("return schema mismatch")
    if manifest.get("sweep_id") != config.sweep_id:
        failures.append("return sweep id mismatch")
    if manifest.get("attempt_id") != attempt:
        failures.append("return attempt id mismatch")
    if manifest.get("source_git_commit") != source_commit:
        failures.append("return source commit mismatch")
    if manifest.get("cell_count") != 16:
        failures.append("return must contain exactly 16 cells")
    try:
        expected_files, expected_cells = _collect_attempt(root, attempt, config, source_commit)
    except (OSError, ValueError, KeyError, SweepReturnError) as error:
        failures.append(f"cannot reconstruct selected attempt: {error}")
        return failures
    if manifest.get("cells") != expected_cells:
        failures.append("return cell identity/status contract mismatch")
    expected = {path.relative_to(root).as_posix(): category for path, category in expected_files}
    entries = manifest.get("files")
    if not isinstance(entries, list):
        failures.append("return files must be a list")
        entries = []
    actual: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is invalid")
            continue
        relative = entry.get("path")
        category = entry.get("category")
        if not isinstance(relative, str) or not isinstance(category, str):
            failures.append(f"files[{index}] has invalid path/category")
            continue
        if relative in actual:
            failures.append(f"duplicate returned path: {relative}")
            continue
        actual[relative] = category
        counts[category] += 1
        path = root / relative
        if path.is_symlink():
            failures.append(f"returned symlink is forbidden: {relative}")
            continue
        if not path.is_file():
            failures.append(f"returned file is missing: {relative}")
            continue
        if entry.get("size_bytes") != path.stat().st_size:
            failures.append(f"returned size mismatch: {relative}")
        if entry.get("sha256") != sha256_file(path):
            failures.append(f"returned hash mismatch: {relative}")
    if set(actual) != set(expected):
        failures.append(
            "return exact inventory mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for relative in set(actual) & set(expected):
        if actual[relative] != expected[relative]:
            failures.append(f"return category mismatch: {relative}")
    if manifest.get("category_counts") != dict(sorted(counts.items())):
        failures.append("return category_counts mismatch")
    if manifest.get("file_count") != len(entries):
        failures.append("return file_count mismatch")
    if manifest.get("total_size_bytes") != sum(
        entry.get("size_bytes", 0) for entry in entries if isinstance(entry, dict)
    ):
        failures.append("return total_size_bytes mismatch")
    if manifest.get("verification") != {"algorithm": "sha256", "status": "PASS"}:
        failures.append("return verification declaration mismatch")
    return failures


def _collect_attempt(
    root: Path,
    attempt_id: str,
    config: SweepConfig,
    source_commit: str,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    attempt_root = root / "attempts" / attempt_id
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise SweepReturnError(f"selected durable attempt is missing: {attempt_root}")
    expected_tasks = {str(cell.index) for cell in config.cells}
    actual_tasks = {path.name for path in attempt_root.iterdir()}
    if actual_tasks != expected_tasks:
        raise SweepReturnError(
            f"selected attempt task inventory mismatch: expected={sorted(expected_tasks)}, "
            f"actual={sorted(actual_tasks)}"
        )
    files: list[tuple[Path, str]] = []
    cells: dict[str, Any] = {}
    for cell in config.cells:
        task_root = attempt_root / str(cell.index)
        if task_root.is_symlink() or not task_root.is_dir():
            raise SweepReturnError(f"invalid task directory: {task_root}")
        runs = {path.name for path in task_root.iterdir()}
        if runs != {cell.run_id}:
            raise SweepReturnError(
                f"task {cell.index} run inventory mismatch: expected={cell.run_id}, actual={runs}"
            )
        run_root = task_root / cell.run_id
        if run_root.is_symlink():
            raise SweepReturnError(f"run root may not be a symlink: {run_root}")
        completion = run_root / "status" / "completion.json"
        failure = run_root / "status" / "failure.json"
        if not completion.is_file() or failure.exists():
            raise SweepReturnError(
                f"cell {cell.run_id} is not uniquely completed; partial/failed sweeps cannot return"
            )
        status = _load_status(completion, cell, attempt_id, source_commit)
        required = {
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "logs/train_log.json",
            "metrics/open_loop.json",
            "resolved_config.json",
            "environment/environment_inventory.json",
            "environment/requirements.lock",
            "environment/preflight_report.json",
            "slurm/stdout.log",
            "slurm/stderr.log",
            "status/completion.json",
            "wandb/publication_report.json",
        }
        for relative in required:
            if not (run_root / relative).is_file():
                raise SweepReturnError(f"completed cell {cell.run_id} missing {relative}")
        publication = json.loads((run_root / "wandb/publication_report.json").read_text())
        if (
            publication.get("destination_contains_symlinks") is not False
            or publication.get("destination_symlink_count") != 0
        ):
            raise SweepReturnError(f"cell {cell.run_id} W&B publication is not symlink-free")
        for path in sorted(run_root.rglob("*")):
            if path.is_symlink():
                raise SweepReturnError(f"symlinks are forbidden in returned results: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(run_root)
            if relative != Path("resolved_config.json") and (
                not relative.parts or relative.parts[0] not in ALLOWED_TOP_LEVEL
            ):
                raise SweepReturnError(f"unexpected returned artifact: {path}")
            if path.suffix.lower() in TEXT_SUFFIXES:
                problems = secret_problems(path.relative_to(root).as_posix(), path.read_bytes())
                if problems:
                    raise SweepReturnError("; ".join(problems))
            files.append((path, _category(relative)))
        cells[cell.run_id] = {
            "index": cell.index,
            "policy": cell.policy,
            "space": cell.space,
            "view_id": cell.view_id,
            "status": status["status"],
            "exit_code": status["exit_code"],
            "attempt": status["attempt"],
        }
    if len(cells) != 16:
        raise SweepReturnError("selected attempt does not contain exactly 16 unique cells")
    return files, cells


def _load_status(
    path: Path, cell: SweepCell, attempt_id: str, source_commit: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected_attempt = {
        "slurm_array_job_id": attempt_id,
        "slurm_array_task_id": str(cell.index),
        "run_id": cell.run_id,
    }
    expected = {
        "schema": STATUS_SCHEMA,
        "status": "COMPLETED",
        "run_id": cell.run_id,
        "policy": cell.policy,
        "space": cell.space,
        "view_id": cell.view_id,
        "exit_code": 0,
        "source_git_commit": source_commit,
        "attempt": expected_attempt,
    }
    if payload != expected:
        raise SweepReturnError(f"cell completion status contract mismatch: {path}")
    return payload


def verify_sweep_checkpoints(
    results_root: str | Path,
    *,
    attempt_id: str,
    config: SweepConfig,
    transfer_manifest: dict[str, Any],
    loaders: dict[str, Callable[[Path], Any]] | None = None,
) -> dict[str, str]:
    root = Path(results_root).resolve()
    attempt = _attempt_id(attempt_id)
    source_commit = _transfer_source_commit(transfer_manifest, config)
    active_loaders = loaders or _checkpoint_loaders()
    statuses: dict[str, str] = {}
    for cell in config.cells:
        path = root / "attempts" / attempt / str(cell.index) / cell.run_id / "checkpoints/best.pt"
        loaded = active_loaders[cell.policy](path)
        dataset_cfg = loaded.config.get("dataset") or {}
        expected_dataset = {
            "task": config.dataset.task,
            "space": cell.space,
            "version": config.dataset.master_version,
            "view_id": cell.view_id,
            "obs_preset": config.dataset.obs_preset,
        }
        if {name: dataset_cfg.get(name) for name in expected_dataset} != expected_dataset:
            raise SweepReturnError(f"returned checkpoint dataset config mismatch: {cell.run_id}")
        provenance = loaded.provenance
        norm = transfer_manifest["dataset"]["normalization_artifacts"][
            f"{cell.space}:{cell.view_id}"
        ]
        view = transfer_manifest["dataset"]["views"][cell.view_id]
        splits = view["splits"]
        split_fingerprint = hashlib.sha256(
            json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected_provenance = {
            "master_dataset_fingerprint_sha256": transfer_manifest["dataset"][
                "spaces"
            ][cell.space]["dataset_fingerprint_sha256"],
            "view_id": cell.view_id,
            "view_fingerprint_sha256": view["view_fingerprint_sha256"],
            "split_episode_ids": splits,
            "split_counts": {name: len(splits[name]) for name in ("train", "val", "test")},
            "split_fingerprint_sha256": split_fingerprint,
            "normalization_sha256": norm["sha256"],
            "normalization_fingerprint_sha256": norm[
                "normalization_fingerprint_sha256"
            ],
            "source_git_commit": source_commit,
            "action_space": cell.space,
            "obs_preset": config.dataset.obs_preset,
        }
        for name, expected in expected_provenance.items():
            if provenance.get(name) != expected:
                raise SweepReturnError(
                    f"returned checkpoint provenance mismatch for {cell.run_id}: {name}"
                )
        statuses[cell.run_id] = "CPU_LOAD_PASS"
    return statuses


def return_file_list(manifest: dict[str, Any]) -> list[str]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise SweepReturnError("return manifest files are missing")
    paths = sorted(entry["path"] for entry in entries)
    if len(paths) != len(set(paths)):
        raise SweepReturnError("return manifest files are duplicated")
    return paths


def return_rsync_template() -> str:
    return (
        "rsync -avP --partial --checksum "
        "--files-from=:<remote_results_root>/.sweep_return/attempts/"
        "<SLURM_ARRAY_JOB_ID>/return-files.txt "
        "<user>@<host>:<remote_results_root>/ <local_return_root>/"
    )


def write_return_artifacts(
    results_root: str | Path, manifest: dict[str, Any]
) -> tuple[Path, Path, Path]:
    root = Path(results_root).resolve()
    attempt = _attempt_id(str(manifest.get("attempt_id")))
    directory = root / ".sweep_return" / "attempts" / attempt
    if directory.exists():
        raise SweepReturnError(f"return package already exists: {directory}")
    directory.mkdir(parents=True)
    manifest_path = directory / "return_manifest.json"
    files_path = directory / "return-files.txt"
    command_path = directory / "return-rsync-command.txt"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(files_path, "\n".join(return_file_list(manifest)) + "\n")
    _atomic_write(command_path, return_rsync_template() + "\n")
    return manifest_path, files_path, command_path


def _transfer_source_commit(manifest: dict[str, Any], config: SweepConfig) -> str:
    if manifest.get("sweep_id") != config.sweep_id:
        raise SweepReturnError("transfer manifest sweep ID mismatch")
    commit = (manifest.get("source_git") or {}).get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SweepReturnError("transfer manifest source commit is invalid")
    return commit


def _attempt_id(value: str) -> str:
    if not isinstance(value, str) or ATTEMPT_RE.fullmatch(value) is None:
        raise SweepReturnError("attempt_id must be an explicit numeric array job ID")
    return value


def _category(relative: Path) -> str:
    if relative == Path("resolved_config.json"):
        return "resolved_config"
    return {
        "checkpoints": "checkpoint",
        "environment": "environment",
        "logs": "train_log",
        "metrics": "open_loop_metrics",
        "plots": "open_loop_plot",
        "slurm": "slurm_log",
        "status": "status",
        "wandb": "wandb",
    }[relative.parts[0]]


def _checkpoint_loaders() -> dict[str, Callable[[Path], Any]]:
    from alexdoor_xas.policies.act.checkpoint import load_checkpoint as load_act
    from alexdoor_xas.policies.diffusion.checkpoint import load_checkpoint as load_diffusion

    return {
        "act": lambda path: load_act(path, map_location="cpu"),
        "diffusion": lambda path: load_diffusion(path, map_location="cpu"),
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "RETURN_SCHEMA",
    "STATUS_SCHEMA",
    "SweepReturnError",
    "build_sweep_return_manifest",
    "return_file_list",
    "return_rsync_template",
    "verify_sweep_checkpoints",
    "verify_sweep_return_manifest",
    "write_return_artifacts",
]
