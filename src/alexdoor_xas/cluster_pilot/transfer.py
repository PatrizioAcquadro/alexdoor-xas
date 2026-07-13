"""Deterministic, fail-closed transfer inventory for the Gilbreth pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alexdoor_xas.dataset import load_split_payload, split_fingerprint
from alexdoor_xas.policies.common.data import PolicyDataError, load_policy_data

from .config import EXPECTED_SPACES, PilotConfig

TRANSFER_SCHEMA = "alexdoor_xas.cluster_pilot_transfer_manifest.v1"
DEFAULT_OUTPUT_DIR = Path("outputs/cluster_pilot_n50")
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "pilot_transfer_manifest.json"
DEFAULT_FILE_LIST_PATH = DEFAULT_OUTPUT_DIR / "rsync-files.txt"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_ASSIGNMENT_RE = re.compile(
    rb"(?i)(api[_-]?key|password|passwd|secret|token|credential)\s*[:=]\s*"
    rb"[^\s\"'<>{}$][^\r\n]*"
)
AUTHENTICATED_URL_RE = re.compile(rb"(?i)https?://[^/@\s:]+:[^/@\s]+@")
PRIVATE_KEY_MARKER = b"-----BEGIN " + b"PRIVATE KEY-----"
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


class PilotTransferError(ValueError):
    """Raised when the transfer package cannot be proven exact and reproducible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return the exact commit, branch, timestamp, and cleanliness of a checkout."""
    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    commit_time = _git(root, "show", "-s", "--format=%cI", "HEAD")
    return {
        "commit": commit,
        "branch": branch or None,
        "detached": not bool(branch),
        "clean_tree": not bool(status),
        "commit_time": commit_time,
    }


def build_transfer_manifest(
    repo_root: str | Path,
    config: PilotConfig,
    *,
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for the exact non-Isaac pilot inputs."""
    root = Path(repo_root).resolve()
    state = dict(source_state) if source_state is not None else git_state(root)
    _validate_source_state(state, require_clean=True)
    files, dataset = _collect_contract(
        root,
        config,
        require_tracked=source_state is None,
    )
    entries = [_file_entry(root, path, category) for path, category in files]
    entries.sort(key=lambda entry: entry["path"])
    manifest = {
        "schema": TRANSFER_SCHEMA,
        "pilot_config_schema": config.schema,
        "pilot_id": config.pilot_id,
        "created_utc": state["commit_time"],
        "source_git": state,
        "dataset": dataset,
        "files": entries,
        "category_counts": dict(
            sorted(Counter(entry["category"] for entry in entries).items())
        ),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "verification": {"algorithm": "sha256", "status": "PASS"},
    }
    failures = verify_transfer_manifest(
        manifest,
        root,
        config,
        source_state=state,
        require_tracked=source_state is None,
    )
    if failures:
        raise PilotTransferError(
            "pilot transfer manifest failed self-verification: " + "; ".join(failures)
        )
    return manifest


def verify_transfer_manifest(
    manifest: dict[str, Any],
    repo_root: str | Path,
    config: PilotConfig,
    *,
    source_state: dict[str, Any] | None = None,
    require_tracked: bool = False,
) -> list[str]:
    """Return every source, inventory, fingerprint, secret, size, and hash failure."""
    failures: list[str] = []
    root = Path(repo_root).resolve()
    state = dict(source_state) if source_state is not None else git_state(root)
    try:
        _validate_source_state(state, require_clean=True)
    except PilotTransferError as error:
        failures.append(str(error))

    if manifest.get("schema") != TRANSFER_SCHEMA:
        failures.append(f"schema must be {TRANSFER_SCHEMA!r}")
    if manifest.get("pilot_config_schema") != config.schema:
        failures.append("pilot config schema mismatch")
    if manifest.get("pilot_id") != config.pilot_id:
        failures.append("pilot id mismatch")
    declared_source = manifest.get("source_git")
    if not isinstance(declared_source, dict):
        failures.append("source_git is missing")
        declared_source = {}
    if declared_source.get("commit") != state.get("commit"):
        failures.append(
            "source commit mismatch: "
            f"manifest={declared_source.get('commit')!r}, checkout={state.get('commit')!r}"
        )
    if declared_source.get("clean_tree") is not True:
        failures.append("manifest source_git.clean_tree must be true")
    if manifest.get("created_utc") != declared_source.get("commit_time"):
        failures.append("created_utc must equal the deterministic source commit timestamp")

    try:
        expected_files, expected_dataset = _collect_contract(
            root,
            config,
            require_tracked=require_tracked,
        )
    except (OSError, ValueError, PolicyDataError, PilotTransferError) as error:
        failures.append(f"cannot reconstruct expected pilot contract: {error}")
        return failures

    declared_dataset = manifest.get("dataset")
    if not isinstance(declared_dataset, dict):
        failures.append("dataset contract is missing")
    else:
        if declared_dataset.get("counts") != expected_dataset["counts"]:
            failures.append("dataset counts mismatch; expected exactly 50 total and 38/6/6")
        if (
            declared_dataset.get("split_fingerprint_sha256")
            != expected_dataset["split_fingerprint_sha256"]
        ):
            failures.append("split fingerprint mismatch")
        declared_spaces = declared_dataset.get("spaces")
        if not isinstance(declared_spaces, dict):
            failures.append("dataset spaces contract is missing")
        else:
            for space in EXPECTED_SPACES:
                declared = declared_spaces.get(space)
                expected = expected_dataset["spaces"][space]
                if not isinstance(declared, dict):
                    failures.append(f"dataset space contract missing: {space}")
                    continue
                if (
                    declared.get("dataset_fingerprint_sha256")
                    != expected["dataset_fingerprint_sha256"]
                ):
                    failures.append(f"dataset fingerprint mismatch for {space}")
                if declared.get("obs_preset") != expected["obs_preset"]:
                    failures.append(f"observation preset mismatch for {space}")
                if declared.get("episode_ids") != expected["episode_ids"]:
                    failures.append(f"episode inventory mismatch for {space}")

    expected_categories = {
        _relative(root, path): category for path, category in expected_files
    }
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        failures.append("files must be a non-empty list")
        entries = []
    actual_categories: dict[str, str] = {}
    actual_counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is not an object")
            continue
        relative = entry.get("path")
        category = entry.get("category")
        if not isinstance(relative, str) or not relative:
            failures.append(f"files[{index}] has an invalid path")
            continue
        if not isinstance(category, str) or not category:
            failures.append(f"files[{index}] has an invalid category")
            continue
        forbidden = _forbidden_path_problem(relative)
        if forbidden:
            failures.append(forbidden)
        if relative in actual_categories:
            failures.append(f"duplicate path in manifest: {relative}")
            continue
        actual_categories[relative] = category
        actual_counts[category] += 1
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        size = entry.get("size_bytes")
        if size != path.stat().st_size:
            failures.append(f"size mismatch: {relative}")
        declared_hash = entry.get("sha256")
        if not isinstance(declared_hash, str) or SHA256_RE.fullmatch(declared_hash) is None:
            failures.append(f"malformed sha256: {relative}")
        elif declared_hash != sha256_file(path):
            failures.append(f"hash mismatch: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            failures.extend(secret_problems(relative, path.read_bytes()))

    missing = sorted(set(expected_categories) - set(actual_categories))
    unexpected = sorted(set(actual_categories) - set(expected_categories))
    if missing:
        failures.append(f"missing expected files: {missing}")
    if unexpected:
        failures.append(f"unexpected or forbidden files: {unexpected}")
    for relative in sorted(set(expected_categories) & set(actual_categories)):
        if actual_categories[relative] != expected_categories[relative]:
            failures.append(
                f"category mismatch for {relative}: "
                f"{actual_categories[relative]!r} != {expected_categories[relative]!r}"
            )
    if manifest.get("category_counts") != dict(sorted(actual_counts.items())):
        failures.append("category_counts does not match files")
    if manifest.get("file_count") != len(entries):
        failures.append("file_count does not match files")
    if manifest.get("total_size_bytes") != sum(
        entry.get("size_bytes", 0) for entry in entries if isinstance(entry, dict)
    ):
        failures.append("total_size_bytes does not match files")
    verification = manifest.get("verification")
    if verification != {"algorithm": "sha256", "status": "PASS"}:
        failures.append("verification declaration must be sha256/PASS")
    return failures


def pilot_rsync_file_list(manifest: dict[str, Any]) -> list[str]:
    """Return the deterministic repository-relative transfer list."""
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise PilotTransferError("manifest files are missing")
    paths = sorted(
        entry["path"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    if len(paths) != len(entries) or len(paths) != len(set(paths)):
        raise PilotTransferError("manifest files are malformed or duplicated")
    return [*paths, DEFAULT_MANIFEST_PATH.as_posix(), DEFAULT_FILE_LIST_PATH.as_posix()]


def pilot_rsync_template() -> str:
    """Exact checksum-based outbound rsync template; placeholders remain literal."""
    return (
        "rsync -avP --partial --checksum "
        f"--files-from={DEFAULT_FILE_LIST_PATH.as_posix()} "
        "./ <user>@<host>:<remote_root>/"
    )


def write_transfer_artifacts(
    repo_root: str | Path,
    manifest: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """Atomically publish the ignored manifest, file list, and command template."""
    root = Path(repo_root).resolve()
    directory = (root / output_dir).resolve()
    directory.relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / DEFAULT_MANIFEST_PATH.name
    files_path = directory / DEFAULT_FILE_LIST_PATH.name
    command_path = directory / "rsync-command.txt"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(files_path, "\n".join(pilot_rsync_file_list(manifest)) + "\n")
    _atomic_write(command_path, pilot_rsync_template() + "\n")
    return manifest_path, files_path, command_path


def secret_problems(relative: str, content: bytes) -> list[str]:
    """Return credential/secret problems without ever echoing the sensitive value."""
    problems: list[str] = []
    lowered_parts = {part.lower() for part in Path(relative).parts}
    sensitive_names = {".env", ".netrc", "credentials", "id_rsa", "id_ed25519"}
    if lowered_parts & sensitive_names or any("secret" in part for part in lowered_parts):
        problems.append(f"secret-bearing path is forbidden: {relative}")
    if PRIVATE_KEY_MARKER in content:
        problems.append(f"private key material is forbidden: {relative}")
    if AUTHENTICATED_URL_RE.search(content):
        problems.append(f"credential-bearing URL is forbidden: {relative}")
    if SECRET_ASSIGNMENT_RE.search(content):
        problems.append(f"credential assignment is forbidden: {relative}")
    return problems


def _collect_contract(
    root: Path,
    config: PilotConfig,
    *,
    require_tracked: bool,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    split_path = (root / config.source_dataset.shared_split).resolve()
    _within(root, split_path)
    if not split_path.is_file():
        raise PilotTransferError(f"shared split is missing: {_relative(root, split_path)}")
    split_payload = load_split_payload(split_path)
    splits = {name: list(split_payload["splits"][name]) for name in ("train", "val", "test")}
    counts = {name: len(ids) for name, ids in splits.items()}
    counts["total"] = sum(counts.values())
    ordered_counts = {
        "total": counts["total"],
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
    }
    if ordered_counts != config.source_dataset.counts:
        raise PilotTransferError(
            f"split counts must be exactly {config.source_dataset.counts}, got {ordered_counts}"
        )
    all_ids = [episode_id for name in ("train", "val", "test") for episode_id in splits[name]]
    if len(all_ids) != len(set(all_ids)):
        raise PilotTransferError("shared split contains duplicate or overlapping episode IDs")
    computed_split_fingerprint = split_fingerprint(splits)
    if split_payload.get("split_fingerprint_sha256") != computed_split_fingerprint:
        raise PilotTransferError("shared split fingerprint is stale")
    if split_payload.get("n_episodes") != config.source_dataset.counts["total"]:
        raise PilotTransferError("shared split n_episodes is stale")

    inventory: list[tuple[Path, str]] = [(split_path, "split")]
    dataset_spaces: dict[str, Any] = {}
    expected_episode_ids = sorted(all_ids)
    prefixes = [episode_id.split("-", 1)[0] for episode_id in expected_episode_ids]
    if len(prefixes) != len(set(prefixes)):
        raise PilotTransferError("episode UUID prefixes are not unique")

    for space in EXPECTED_SPACES:
        dataset_dir = (root / config.source_dataset.spaces[space]).resolve()
        _within(root, dataset_dir)
        if not dataset_dir.is_dir():
            raise PilotTransferError(
                f"dataset directory is missing: {_relative(root, dataset_dir)}"
            )
        expected_paths: set[Path] = set()
        for episode_id, prefix in zip(expected_episode_ids, prefixes, strict=True):
            hdf5 = dataset_dir / f"episode_{prefix}.hdf5"
            sidecar = dataset_dir / f"episode_{prefix}.meta.json"
            for path in (hdf5, sidecar):
                if not path.is_file():
                    raise PilotTransferError(
                        f"missing split-derived episode input: {_relative(root, path)}"
                    )
                expected_paths.add(path)
                inventory.append((path, "dataset_episode"))
            try:
                sidecar_payload = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise PilotTransferError(f"invalid sidecar {sidecar}: {error}") from error
            meta = sidecar_payload.get("meta")
            if not isinstance(meta, dict) or meta.get("episode_id") != episode_id:
                raise PilotTransferError(f"stale episode sidecar: {_relative(root, sidecar)}")
            if meta.get("task") != config.source_dataset.task or meta.get("action_space") != space:
                raise PilotTransferError(
                    f"sidecar task/action-space mismatch: {_relative(root, sidecar)}"
                )

        metadata_paths = [
            dataset_dir / "meta.json",
            dataset_dir / "manifest.json",
            dataset_dir / "norm_stats.json",
        ]
        for path in metadata_paths:
            if not path.is_file():
                raise PilotTransferError(f"missing dataset metadata: {_relative(root, path)}")
            expected_paths.add(path)
            inventory.append((path, "dataset_metadata"))
        actual_paths = {path for path in dataset_dir.iterdir() if path.is_file()}
        actual_directories = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
        if actual_directories:
            raise PilotTransferError(
                f"unexpected directories in {_relative(root, dataset_dir)}: {actual_directories}"
            )
        missing = sorted(_relative(root, path) for path in expected_paths - actual_paths)
        extra = sorted(_relative(root, path) for path in actual_paths - expected_paths)
        if missing or extra:
            raise PilotTransferError(
                f"dataset inventory mismatch for {space}: missing={missing}, extra={extra}"
            )

        meta_payload = json.loads((dataset_dir / "meta.json").read_text())
        if (
            meta_payload.get("task") != config.source_dataset.task
            or meta_payload.get("action_space") != space
            or meta_payload.get("n_episodes") != config.source_dataset.counts["total"]
        ):
            raise PilotTransferError(f"dataset meta.json contract mismatch for {space}")
        dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text())
        manifest_ids = sorted(
            entry.get("episode_id")
            for entry in dataset_manifest.get("episodes", [])
            if isinstance(entry, dict)
        )
        if manifest_ids != expected_episode_ids:
            raise PilotTransferError(f"dataset manifest episode inventory is stale for {space}")

        policy_cfg = SimpleNamespace(
            task=config.source_dataset.task,
            space=space,
            version=config.source_dataset.version,
            obs_preset=config.source_dataset.obs_preset,
        )
        data = load_policy_data(policy_cfg, datasets_root=root / "datasets")
        if sorted(data.dataset.episode_ids) != expected_episode_ids:
            raise PilotTransferError(f"loaded dataset episode inventory is stale for {space}")
        if (
            list(data.train_ids) != splits["train"]
            or list(data.val_ids) != splits["val"]
            or list(data.test_ids) != splits["test"]
        ):
            raise PilotTransferError(f"loaded split order/content mismatch for {space}")
        fingerprint = data.stats.dataset_fingerprint
        if SHA256_RE.fullmatch(fingerprint) is None:
            raise PilotTransferError(f"invalid computed dataset fingerprint for {space}")
        dataset_spaces[space] = {
            "path": config.source_dataset.spaces[space],
            "obs_preset": config.source_dataset.obs_preset,
            "dataset_fingerprint_sha256": fingerprint,
            "official_norm_stats_obs_preset": json.loads(
                (dataset_dir / "norm_stats.json").read_text()
            ).get("obs_preset"),
            "episode_ids": expected_episode_ids,
        }

    for relative in config.tracked_transfer_files:
        path = (root / relative).resolve()
        _within(root, path)
        if not path.is_file():
            raise PilotTransferError(f"tracked pilot source is missing: {relative}")
        if require_tracked and not _is_tracked(root, relative):
            raise PilotTransferError(f"pilot source is not tracked by Git: {relative}")
        problems = secret_problems(relative, path.read_bytes())
        if problems:
            raise PilotTransferError("; ".join(problems))
        inventory.append((path, "pilot_source"))

    relative_paths = [_relative(root, path) for path, _ in inventory]
    if len(relative_paths) != len(set(relative_paths)):
        raise PilotTransferError("pilot inventory contains duplicate paths")
    dataset = {
        "task": config.source_dataset.task,
        "version": config.source_dataset.version,
        "obs_preset": config.source_dataset.obs_preset,
        "counts": config.source_dataset.counts,
        "split_path": config.source_dataset.shared_split,
        "split_fingerprint_sha256": computed_split_fingerprint,
        "split_episode_ids": splits,
        "spaces": dataset_spaces,
    }
    return inventory, dataset


def _file_entry(root: Path, path: Path, category: str) -> dict[str, Any]:
    return {
        "category": category,
        "path": _relative(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_source_state(state: dict[str, Any], *, require_clean: bool) -> None:
    commit = state.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PilotTransferError("source Git commit must be a full 40-character SHA-1")
    if require_clean and state.get("clean_tree") is not True:
        raise PilotTransferError("source tree must be clean before building or verifying the pilot")
    commit_time = state.get("commit_time")
    if not isinstance(commit_time, str) or not commit_time:
        raise PilotTransferError("source Git commit timestamp is missing")


def _forbidden_path_problem(relative: str) -> str | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        return f"forbidden non-relative path: {relative}"
    forbidden_parts = {"A1_joint_delta", "A4_obj_centric_chunk", ".git", "checkpoints"}
    if forbidden_parts & set(path.parts):
        return f"forbidden pilot input path: {relative}"
    if path.parts and path.parts[0] == "outputs":
        return f"forbidden prior/generated output path: {relative}"
    lowered = relative.lower()
    if any(token in lowered for token in ("isaacsim", "isaac_lab", "isaaclab.sh")):
        return f"forbidden Isaac input path: {relative}"
    return None


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _within(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PilotTransferError(f"path escapes repository: {path}") from error


def _git(root: Path, *args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode and not allow_failure:
        raise PilotTransferError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_tracked(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


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
    "PilotTransferError",
    "TRANSFER_SCHEMA",
    "build_transfer_manifest",
    "git_state",
    "pilot_rsync_file_list",
    "pilot_rsync_template",
    "secret_problems",
    "sha256_file",
    "verify_transfer_manifest",
    "write_transfer_artifacts",
]
