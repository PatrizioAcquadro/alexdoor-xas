"""Durable result inventory and Ubuntu-side verification for the pilot."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PilotCell, PilotConfig
from .transfer import TRANSFER_SCHEMA, secret_problems, sha256_file

RETURN_SCHEMA = "alexdoor_xas.cluster_pilot_return_manifest.v2"
RETURN_ARTIFACT_DIR = Path(".pilot_return")
RETURN_ATTEMPTS_DIR = RETURN_ARTIFACT_DIR / "attempts"
RETURN_MANIFEST_NAME = "return_manifest.json"
RETURN_FILE_LIST_NAME = "return-files.txt"
ATTEMPT_ID_RE = re.compile(r"^[0-9]+$")
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
TEXT_SUFFIXES = {".json", ".log", ".txt", ".yaml", ".yml"}


class ReturnManifestError(ValueError):
    """Raised when durable pilot results are missing, unsafe, or inconsistent."""


def build_return_manifest(
    results_root: str | Path,
    config: PilotConfig,
    transfer_manifest: dict[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    """Inventory one explicitly selected durable Slurm-array attempt with hashes."""
    root = Path(results_root).resolve()
    selected_attempt = _validate_attempt_id(attempt_id)
    if not root.is_dir():
        raise ReturnManifestError(f"durable results root does not exist: {root}")
    if transfer_manifest.get("schema") != TRANSFER_SCHEMA:
        raise ReturnManifestError("source transfer manifest schema is invalid")
    source_commit = transfer_manifest.get("source_git", {}).get("commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ReturnManifestError("source transfer commit is invalid")
    files, cells = _collect_return_files(
        root,
        config,
        attempt_id=selected_attempt,
        expected_source_commit=source_commit,
    )
    entries = [
        {
            "category": category,
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, category in files
    ]
    entries.sort(key=lambda entry: entry["path"])
    return {
        "schema": RETURN_SCHEMA,
        "pilot_config_schema": config.schema,
        "pilot_id": config.pilot_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_transfer_schema": transfer_manifest["schema"],
        "source_git_commit": source_commit,
        "provenance": _return_provenance(selected_attempt, config, source_commit),
        "cells": cells,
        "files": entries,
        "category_counts": dict(
            sorted(Counter(entry["category"] for entry in entries).items())
        ),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "verification": {"algorithm": "sha256", "status": "PASS"},
    }


def verify_return_manifest(
    manifest: dict[str, Any],
    results_root: str | Path,
    config: PilotConfig,
    *,
    attempt_id: str,
) -> list[str]:
    """Return all missing, extra, status, size, secret, and hash failures."""
    failures: list[str] = []
    root = Path(results_root).resolve()
    try:
        selected_attempt = _validate_attempt_id(attempt_id)
    except ReturnManifestError as error:
        return [str(error)]
    if manifest.get("schema") != RETURN_SCHEMA:
        failures.append(f"schema must be {RETURN_SCHEMA!r}")
    if manifest.get("pilot_config_schema") != config.schema:
        failures.append("pilot config schema mismatch")
    if manifest.get("pilot_id") != config.pilot_id:
        failures.append("pilot id mismatch")
    source_commit = manifest.get("source_git_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        failures.append("source Git commit is invalid")
        return failures
    provenance = manifest.get("provenance")
    declared_attempt = (
        provenance.get("slurm_array_job_id") if isinstance(provenance, dict) else None
    )
    if declared_attempt != selected_attempt:
        failures.append(
            "selected attempt mismatch: "
            f"manifest={declared_attempt!r}, requested={selected_attempt!r}"
        )
        return failures
    expected_provenance = _return_provenance(selected_attempt, config, source_commit)
    if provenance != expected_provenance:
        failures.append("return provenance mismatch")
    try:
        expected_files, expected_cells = _collect_return_files(
            root,
            config,
            attempt_id=selected_attempt,
            expected_source_commit=source_commit,
        )
    except ReturnManifestError as error:
        failures.append(str(error))
        return failures
    if manifest.get("cells") != expected_cells:
        failures.append("cell status/inventory mismatch")

    expected_categories = {
        path.relative_to(root).as_posix(): category for path, category in expected_files
    }
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        failures.append("files must be a non-empty list")
        entries = []
    actual_categories: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is not an object")
            continue
        relative = entry.get("path")
        category = entry.get("category")
        if not isinstance(relative, str) or not relative:
            failures.append(f"files[{index}] has invalid path")
            continue
        if not isinstance(category, str) or not category:
            failures.append(f"files[{index}] has invalid category")
            continue
        if relative in actual_categories:
            failures.append(f"duplicate return path: {relative}")
            continue
        actual_categories[relative] = category
        counts[category] += 1
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"return path escapes results root: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing returned file: {relative}")
            continue
        if entry.get("size_bytes") != path.stat().st_size:
            failures.append(f"size mismatch: {relative}")
        if entry.get("sha256") != sha256_file(path):
            failures.append(f"hash mismatch: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            failures.extend(secret_problems(relative, path.read_bytes()))
    missing = sorted(set(expected_categories) - set(actual_categories))
    extra = sorted(set(actual_categories) - set(expected_categories))
    if missing:
        failures.append(f"missing returned artifacts: {missing}")
    if extra:
        failures.append(f"unexpected returned artifacts: {extra}")
    for relative in sorted(set(expected_categories) & set(actual_categories)):
        if actual_categories[relative] != expected_categories[relative]:
            failures.append(f"return category mismatch: {relative}")
    if manifest.get("category_counts") != dict(sorted(counts.items())):
        failures.append("category_counts does not match return files")
    if manifest.get("file_count") != len(entries):
        failures.append("file_count does not match return files")
    if manifest.get("total_size_bytes") != sum(
        entry.get("size_bytes", 0) for entry in entries if isinstance(entry, dict)
    ):
        failures.append("total_size_bytes does not match return files")
    if manifest.get("verification") != {"algorithm": "sha256", "status": "PASS"}:
        failures.append("verification declaration must be sha256/PASS")
    return failures


def return_file_list(manifest: dict[str, Any], attempt_id: str) -> list[str]:
    selected_attempt = _validate_attempt_id(attempt_id)
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("slurm_array_job_id") != selected_attempt
    ):
        raise ReturnManifestError("return manifest does not match the selected attempt")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ReturnManifestError("return manifest files are missing")
    paths = sorted(
        entry["path"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    if len(paths) != len(entries) or len(paths) != len(set(paths)):
        raise ReturnManifestError("return manifest paths are malformed or duplicated")
    return [
        *paths,
        (RETURN_ATTEMPTS_DIR / selected_attempt / RETURN_MANIFEST_NAME).as_posix(),
        (RETURN_ATTEMPTS_DIR / selected_attempt / RETURN_FILE_LIST_NAME).as_posix(),
    ]


def return_rsync_template(attempt_id: str) -> str:
    """Exact checksum-based return rsync template using the remote file list."""
    selected_attempt = _validate_attempt_id(attempt_id)
    return (
        "rsync -avP --partial --checksum "
        "--files-from=:<remote_results_root>/.pilot_return/attempts/"
        f"{selected_attempt}/return-files.txt "
        "<user>@<host>:<remote_results_root>/ <local_return_root>/"
    )


def write_return_artifacts(
    results_root: str | Path,
    manifest: dict[str, Any],
    *,
    attempt_id: str,
) -> tuple[Path, Path, Path]:
    root = Path(results_root).resolve()
    selected_attempt = _validate_attempt_id(attempt_id)
    directory = root / RETURN_ATTEMPTS_DIR / selected_attempt
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / RETURN_MANIFEST_NAME
    files_path = directory / RETURN_FILE_LIST_NAME
    command_path = directory / "return-rsync-command.txt"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(
        files_path, "\n".join(return_file_list(manifest, selected_attempt)) + "\n"
    )
    _atomic_write(command_path, return_rsync_template(selected_attempt) + "\n")
    return manifest_path, files_path, command_path


def verify_return_checkpoints(
    results_root: str | Path,
    config: PilotConfig,
    *,
    attempt_id: str,
    loaders: dict[str, Callable[[Path], Any]] | None = None,
) -> dict[str, str]:
    """Load one best checkpoint per policy on CPU without importing Isaac."""
    root = Path(results_root).resolve()
    selected_attempt = _validate_attempt_id(attempt_id)
    selected_loaders = loaders or _default_checkpoint_loaders()
    statuses: dict[str, str] = {}
    for cell in config.cells:
        loader = selected_loaders.get(cell.policy)
        if loader is None:
            raise ReturnManifestError(f"no checkpoint loader registered for {cell.policy}")
        path = _attempt_run_root(root, selected_attempt, cell) / "checkpoints" / "best.pt"
        if not path.is_file():
            raise ReturnManifestError(f"returned best checkpoint is missing: {path}")
        try:
            loaded = loader(path)
        except Exception as error:  # noqa: BLE001 - normalize safe checkpoint loader failures.
            raise ReturnManifestError(
                f"cannot load {cell.policy} checkpoint {path}: {error}"
            ) from error
        config_payload = getattr(loaded, "config", None)
        if isinstance(config_payload, dict):
            dataset = config_payload.get("dataset")
            run = config_payload.get("run")
            if not isinstance(dataset, dict) or dataset.get("space") != cell.space:
                raise ReturnManifestError(f"checkpoint action-space mismatch for {cell.run_id}")
            if not isinstance(run, dict) or run.get("run_id") != cell.run_id:
                raise ReturnManifestError(f"checkpoint run-id mismatch for {cell.run_id}")
            if dataset.get("obs_preset") != config.source_dataset.obs_preset:
                raise ReturnManifestError(f"checkpoint obs-preset mismatch for {cell.run_id}")
        statuses[cell.run_id] = "PASS"
    return statuses


def _collect_return_files(
    root: Path,
    config: PilotConfig,
    *,
    attempt_id: str,
    expected_source_commit: str | None,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    files: list[tuple[Path, str]] = []
    cells: dict[str, Any] = {}
    attempt_root = root / "attempts" / attempt_id
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise ReturnManifestError(
            f"selected durable attempt is missing or invalid: {attempt_root}"
        )
    expected_tasks = {str(cell.index) for cell in config.cells}
    actual_tasks = {path.name for path in attempt_root.iterdir()}
    if actual_tasks != expected_tasks or any(
        not path.is_dir() or path.is_symlink() for path in attempt_root.iterdir()
    ):
        raise ReturnManifestError(
            "selected durable attempt task inventory mismatch: "
            f"expected={sorted(expected_tasks)}, actual={sorted(actual_tasks)}"
        )
    for cell in config.cells:
        task_root = attempt_root / str(cell.index)
        actual_runs = {path.name for path in task_root.iterdir()}
        if actual_runs != {cell.run_id} or any(
            not path.is_dir() or path.is_symlink() for path in task_root.iterdir()
        ):
            raise ReturnManifestError(
                f"durable result inventory mismatch for task {cell.index}: "
                f"expected={[cell.run_id]!r}, actual={sorted(actual_runs)!r}"
            )
        run_root = _attempt_run_root(root, attempt_id, cell)
        completion = run_root / "status" / "completion.json"
        failure = run_root / "status" / "failure.json"
        if completion.is_file() == failure.is_file():
            raise ReturnManifestError(
                f"{cell.run_id} must contain exactly one completion/failure status"
            )
        status_path = completion if completion.is_file() else failure
        status_payload = _load_status(
            status_path,
            cell,
            attempt_id=attempt_id,
            expected_source_commit=expected_source_commit,
        )
        completed = status_path.name == "completion.json"
        if completed:
            required = [
                "checkpoints/best.pt",
                "checkpoints/last.pt",
                "logs/train_log.json",
                "metrics/open_loop.json",
                "resolved_config.json",
                "environment/environment_inventory.json",
                "environment/requirements.lock",
                "slurm/stdout.log",
                "slurm/stderr.log",
                "status/completion.json",
            ]
            for relative in required:
                path = run_root / relative
                if not path.is_file():
                    raise ReturnManifestError(
                        f"completed cell {cell.run_id} is missing required artifact: {relative}"
                    )
            wandb_files = [path for path in (run_root / "wandb").rglob("*") if path.is_file()]
            if not wandb_files:
                raise ReturnManifestError(
                    f"completed cell {cell.run_id} has no W&B offline directory or ID artifact"
                )
        else:
            for relative in (
                "environment/environment_inventory.json",
                "slurm/stdout.log",
                "slurm/stderr.log",
                "status/failure.json",
            ):
                if not (run_root / relative).is_file():
                    raise ReturnManifestError(
                        f"failed cell {cell.run_id} is missing diagnostic artifact: {relative}"
                    )

        for path in sorted(run_root.rglob("*")):
            if path.is_symlink():
                raise ReturnManifestError(f"symlinks are forbidden in returned results: {path}")
            if not path.is_file():
                continue
            relative_to_run = path.relative_to(run_root)
            allowed_root_file = relative_to_run == Path("resolved_config.json")
            if (
                not relative_to_run.parts
                or (
                    relative_to_run.parts[0] not in ALLOWED_TOP_LEVEL
                    and not allowed_root_file
                )
            ):
                raise ReturnManifestError(f"unexpected returned artifact: {path}")
            if path.suffix.lower() in TEXT_SUFFIXES:
                problems = secret_problems(path.relative_to(root).as_posix(), path.read_bytes())
                if problems:
                    raise ReturnManifestError("; ".join(problems))
            files.append((path, _return_category(relative_to_run)))
        cells[cell.run_id] = {
            "policy": cell.policy,
            "space": cell.space,
            "status": status_payload["status"],
            "exit_code": status_payload["exit_code"],
            "attempt": status_payload["attempt"],
        }
    return files, cells


def _load_status(
    path: Path,
    cell: PilotCell,
    *,
    attempt_id: str,
    expected_source_commit: str | None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReturnManifestError(f"invalid cell status {path}: {error}") from error
    expected_status = "COMPLETED" if path.name == "completion.json" else "FAILED"
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "alexdoor_xas.cluster_pilot_cell_status.v2"
        or payload.get("run_id") != cell.run_id
        or payload.get("policy") != cell.policy
        or payload.get("space") != cell.space
        or payload.get("status") != expected_status
        or not isinstance(payload.get("exit_code"), int)
    ):
        raise ReturnManifestError(f"cell status contract mismatch: {path}")
    expected_attempt = {
        "slurm_array_job_id": attempt_id,
        "slurm_array_task_id": str(cell.index),
        "run_id": cell.run_id,
    }
    if payload.get("attempt") != expected_attempt:
        raise ReturnManifestError(f"cell attempt identity mismatch: {path}")
    if (
        expected_source_commit is not None
        and payload.get("source_git_commit") != expected_source_commit
    ):
        raise ReturnManifestError(f"cell source commit mismatch: {path}")
    if expected_status == "COMPLETED" and payload["exit_code"] != 0:
        raise ReturnManifestError(f"completed status has nonzero exit code: {path}")
    if expected_status == "FAILED" and payload["exit_code"] == 0:
        raise ReturnManifestError(f"failure status has zero exit code: {path}")
    return payload


def _validate_attempt_id(attempt_id: str) -> str:
    if not isinstance(attempt_id, str) or ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise ReturnManifestError("attempt_id must be an explicit numeric Slurm array job ID")
    return attempt_id


def _attempt_run_root(root: Path, attempt_id: str, cell: PilotCell) -> Path:
    return root / "attempts" / attempt_id / str(cell.index) / cell.run_id


def _return_provenance(
    attempt_id: str,
    config: PilotConfig,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "source_git_commit": source_commit,
        "slurm_array_job_id": attempt_id,
        "cells": {
            cell.run_id: {
                "slurm_array_job_id": attempt_id,
                "slurm_array_task_id": str(cell.index),
                "run_id": cell.run_id,
            }
            for cell in config.cells
        },
    }


def _return_category(relative: Path) -> str:
    if relative.name == "resolved_config.json":
        return "resolved_config"
    first = relative.parts[0]
    return {
        "checkpoints": "checkpoint",
        "environment": "environment",
        "logs": "train_log",
        "metrics": "open_loop_metrics",
        "plots": "open_loop_plot",
        "slurm": "slurm_log",
        "status": "status",
        "wandb": "wandb",
    }[first]


def _default_checkpoint_loaders() -> dict[str, Callable[[Path], Any]]:
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
    "ReturnManifestError",
    "build_return_manifest",
    "return_file_list",
    "return_rsync_template",
    "verify_return_checkpoints",
    "verify_return_manifest",
    "write_return_artifacts",
]
