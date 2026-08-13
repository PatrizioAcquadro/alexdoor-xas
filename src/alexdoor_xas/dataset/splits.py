"""Create and load deterministic, leakage-safe dataset splits and views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SPLIT_NAMES = ("train", "val", "test")
_DEFAULT_FRACTIONS = (0.75, 0.125, 0.125)
_DEFAULT_POSE_ID = "default"
_GROUPING_STRATEGY = "content_sha256"
_VIEW_SCHEMA = "alexdoor_xas.dataset_view.v1"


@dataclass(frozen=True)
class SplitEntry:
    """One episode's split-relevant identity (pure data, no file I/O)."""

    episode_id: str
    group_key: str
    pose_id: str = _DEFAULT_POSE_ID


def episode_content_key(record) -> str:
    """Content-equivalence key for one loaded :class:`EpisodeRecord`.

    Hashes only trajectory content: step times, actions, every numeric
    observation array (sorted by key), and the outcome. Two episodes generated
    from different seeds that produced the same rollout hash identically;
    any numeric difference (however small) produces a different key.
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(record.t, dtype=np.float64).tobytes())
    digest.update(b"\0actions\0")
    digest.update(np.asarray(record.actions, dtype=np.float64).tobytes())
    for key in sorted(record.obs):
        digest.update(b"\0obs:" + key.encode() + b"\0")
        digest.update(np.asarray(record.obs[key], dtype=np.float64).tobytes())
    digest.update(b"\0outcome\0")
    digest.update(str(bool(record.success)).encode())
    digest.update(np.asarray([record.final_door_angle], dtype=np.float64).tobytes())
    return digest.hexdigest()


def split_entries(dataset) -> list[SplitEntry]:
    """Build :class:`SplitEntry` rows from an ``EpisodeDataset``."""
    entries = []
    for record in dataset.records:
        pose = record.extras.get("door_pose_id")
        entries.append(
            SplitEntry(
                episode_id=record.episode_id,
                group_key=episode_content_key(record),
                pose_id=str(pose) if pose else _DEFAULT_POSE_ID,
            )
        )
    return entries


def make_grouped_splits(
    entries: list[SplitEntry],
    fractions: tuple[float, float, float] = _DEFAULT_FRACTIONS,
    seed: int = 0,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Assign content groups to pose-stratified train/val/test splits."""
    _check_fractions(fractions)
    ids = [entry.episode_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids are not unique")
    n = len(entries)
    if n < 3:
        raise ValueError(f"need at least 3 episodes to split, got {n}")

    # Group by content key; a group spanning poses would mean two identical
    # trajectories recorded at different door poses — physically impossible
    # for pose-aware content, so treat it as corrupted input.
    groups: dict[str, list[SplitEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.group_key, []).append(entry)
    for key, members in groups.items():
        poses = {member.pose_id for member in members}
        if len(poses) != 1:
            raise ValueError(
                f"content group {key[:12]} spans door poses {sorted(poses)} — "
                "grouping input is inconsistent"
            )

    pose_groups: dict[str, list[str]] = {}
    for key in sorted(groups):
        pose_groups.setdefault(groups[key][0].pose_id, []).append(key)

    n_val_target = max(1, round(fractions[1] * n))
    n_test_target = max(1, round(fractions[2] * n))
    if n - n_val_target - n_test_target < 1:
        raise ValueError(f"fractions {fractions} leave no training episodes for n={n}")

    short = {pose: len(keys) for pose, keys in pose_groups.items() if len(keys) < len(_SPLIT_NAMES)}
    if short:
        raise ValueError(
            "cannot stratify splits by pose: pose(s) "
            + ", ".join(f"{pose!r} ({count} independent groups)" for pose, count in short.items())
            + f" need at least {len(_SPLIT_NAMES)} content-equivalence groups each "
            "(one per split); collect more independent episodes for these poses "
            "or drop them from the dataset"
        )

    rng = np.random.default_rng(seed)
    assigned: dict[str, str] = {}
    counts = {name: 0 for name in _SPLIT_NAMES}

    def assign(key: str, split: str) -> None:
        assigned[key] = split
        counts[split] += len(groups[key])

    # Pose coverage first: one group per pose into test and val. Prefer the
    # smallest groups (random among size ties) so a large duplicate block
    # cannot blow the val/test size targets through a coverage pick.
    remaining: list[str] = []
    for pose in sorted(pose_groups):
        keys = pose_groups[pose]
        tiebreak = rng.permutation(len(keys))
        ordered = sorted(range(len(keys)), key=lambda i: (len(groups[keys[i]]), tiebreak[i]))
        assign(keys[ordered[0]], "test")
        assign(keys[ordered[1]], "val")
        remaining.extend(keys[i] for i in ordered[2:])

    # Fill val/test toward the requested episode counts with whole groups
    # (never split a group); everything else trains.
    order = rng.permutation(len(remaining))
    for index in order:
        key = remaining[index]
        size = len(groups[key])
        if counts["test"] + size <= n_test_target:
            assign(key, "test")
        elif counts["val"] + size <= n_val_target:
            assign(key, "val")
        else:
            assign(key, "train")
    if counts["train"] < 1:
        raise ValueError(
            f"grouping and pose coverage left no training episodes "
            f"(n={n}, groups={len(groups)}, fractions={fractions})"
        )

    splits = {
        name: sorted(
            member.episode_id
            for key, split in assigned.items()
            if split == name
            for member in groups[key]
        )
        for name in _SPLIT_NAMES
    }

    per_pose: dict[str, Any] = {}
    for pose in sorted(pose_groups):
        pose_counts = {name: 0 for name in _SPLIT_NAMES}
        for key in pose_groups[pose]:
            pose_counts[assigned[key]] += len(groups[key])
        per_pose[pose] = {
            "n_groups": len(pose_groups[pose]),
            "n_episodes": sum(len(groups[key]) for key in pose_groups[pose]),
            "episodes_per_split": pose_counts,
        }
    metadata = {
        "strategy": "grouped_pose_stratified",
        "grouping": _GROUPING_STRATEGY,
        "seed": seed,
        "fractions": list(fractions),
        "n_episodes": n,
        "n_groups": len(groups),
        "requested_sizes": {
            "train": n - n_val_target - n_test_target,
            "val": n_val_target,
            "test": n_test_target,
        },
        "actual_sizes": {name: counts[name] for name in _SPLIT_NAMES},
        "size_deviation": {
            "val": counts["val"] - n_val_target,
            "test": counts["test"] - n_test_target,
        },
        "per_pose": per_pose,
        "groups": {
            key: {
                "pose_id": groups[key][0].pose_id,
                "split": assigned[key],
                "episode_ids": sorted(member.episode_id for member in groups[key]),
            }
            for key in sorted(groups)
        },
    }
    return splits, metadata


def _check_fractions(fractions: tuple[float, float, float]) -> None:
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError(f"fractions must be 3 positive values, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")


def assert_no_cross_split_duplicates(
    entries: list[SplitEntry], splits: dict[str, list[str]]
) -> None:
    """Fail if any content-equivalence group spans more than one split."""
    membership = {eid: name for name in _SPLIT_NAMES for eid in splits[name]}
    group_splits: dict[str, set[str]] = {}
    for entry in entries:
        if entry.episode_id not in membership:
            raise ValueError(f"episode {entry.episode_id} is not covered by the splits")
        group_splits.setdefault(entry.group_key, set()).add(membership[entry.episode_id])
    crossing = {key: names for key, names in group_splits.items() if len(names) > 1}
    if crossing:
        raise ValueError(
            "content-equivalent episodes cross split boundaries: "
            + "; ".join(f"group {key[:12]} in {sorted(names)}" for key, names in crossing.items())
        )


def splits_path(datasets_root: str | Path, task: str, version: str) -> Path:
    """Canonical split-file location: per task + version, shared across spaces."""
    return Path(datasets_root) / task / "splits" / f"{version}.json"


def save_splits(
    path: str | Path,
    splits: dict[str, list[str]],
    *,
    fractions: tuple[float, float, float] = _DEFAULT_FRACTIONS,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fractions": list(fractions),
        "seed": seed,
        "n_episodes": sum(len(ids) for ids in splits.values()),
        "splits": {name: splits[name] for name in _SPLIT_NAMES},
    }
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_splits(path: str | Path, episode_ids: list[str] | None = None) -> dict[str, list[str]]:
    """Load a split file; if ``episode_ids`` is given, reject a stale file."""
    target = Path(path)
    splits = _parse_splits(json.loads(target.read_text()), target, "split file")

    all_ids = [eid for ids in splits.values() for eid in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f"split file {target} has overlapping splits")
    if episode_ids is not None and set(all_ids) != set(episode_ids):
        raise ValueError(
            f"split file {target} does not match the dataset episodes — the dataset "
            "was re-exported; regenerate the splits"
        )
    return splits


def load_view_splits(
    path: str | Path,
    *,
    view_id: str,
    master_version: str,
    episode_ids: list[str],
) -> dict[str, list[str]]:
    """Load one retained view and validate its master-dataset membership."""
    target = Path(path)
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != _VIEW_SCHEMA:
        raise ValueError(f"invalid dataset view schema: {target}")
    if payload.get("view_id") != view_id:
        raise ValueError("dataset view ID does not match its path")
    if payload.get("master_version") != master_version:
        raise ValueError("dataset view master version does not match dataset.version")
    splits = _parse_splits(payload, target, "dataset view")
    selected_ids = [episode_id for ids in splits.values() for episode_id in ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("dataset view has overlapping split memberships")
    if not set(selected_ids).issubset(episode_ids):
        raise ValueError("dataset view references episodes absent from the master")
    return splits


def _parse_splits(payload: Any, path: Path, label: str) -> dict[str, list[str]]:
    try:
        values = payload["splits"]
        if not isinstance(values, dict):
            raise TypeError
        splits = {name: values[name] for name in _SPLIT_NAMES}
        if not all(isinstance(ids, list) for ids in splits.values()):
            raise TypeError
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} has invalid splits: {path}") from error
    if not all(
        isinstance(episode_id, str) and episode_id for ids in splits.values() for episode_id in ids
    ):
        raise ValueError(f"{label} split members must be non-empty episode IDs")
    return splits


def _safe_view_id(view_id: str) -> str:
    if not isinstance(view_id, str) or not view_id or "/" in view_id or ".." in view_id:
        raise ValueError("view_id must be a safe single path component")
    return view_id


def view_path(datasets_root: str | Path, task: str, view_id: str) -> Path:
    return Path(datasets_root) / task / "splits" / f"{_safe_view_id(view_id)}.json"
