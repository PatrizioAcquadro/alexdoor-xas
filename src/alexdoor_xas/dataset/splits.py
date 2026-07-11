"""Deterministic train/val/test splits, shared across action spaces (Phase 3.0).

Episode ids are identical across every action-space export of one generation
pass (``data_engine/export.py`` relabels the same episodes in place), so splits
are computed **once per task/version** and shared by A1–A4 — matched-condition
action-space comparisons require it. The split file lives at
``datasets/<task>/splits/<version>.json``; a re-export mints fresh episode ids,
so re-exporting a version means regenerating its splits (same command, same
result for the same seed).

Split contract (post-Phase 3.3 review):

- Episodes are grouped by **trajectory content** (:func:`episode_content_key`):
  fixed-seed replicas of one deterministic rollout are numerically identical,
  and a content-equivalent group must never cross a split boundary (train/test
  leakage). Equivalence is exact byte equality of the recorded trajectory
  arrays — provenance-only fields (episode id, seed, file path, timestamps)
  are excluded, so near-but-not-identical trajectories never collapse.
- Splits are **pose-stratified**: every door pose in the dataset must appear
  in validation and test (and train) whenever it has enough independent
  groups; otherwise split generation fails loudly.
- Fractions are honored as closely as grouping and pose coverage allow; the
  saved metadata records the requested vs. achieved sizes and the grouping so
  the split is auditable after the fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.75, 0.125, 0.125)
DEFAULT_POSE_ID = "default"
"""Pose stratum for episodes recorded without an explicit ``door_pose_id``."""

GROUPING_STRATEGY = "content_sha256"
"""Equivalence identity: sha256 over the episode's recorded trajectory content
(step times, actions, every numeric per-step observation table, outcome).
Episode ids, seeds, file paths, creation times, and split labels are excluded
by construction."""


@dataclass(frozen=True)
class SplitEntry:
    """One episode's split-relevant identity (pure data, no file I/O)."""

    episode_id: str
    group_key: str
    pose_id: str = DEFAULT_POSE_ID


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
                pose_id=str(pose) if pose else DEFAULT_POSE_ID,
            )
        )
    return entries


def make_grouped_splits(
    entries: list[SplitEntry],
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Grouped, pose-stratified, deterministic splits.

    Returns ``(splits, metadata)``. Invariants (all enforced, fail loudly):

    - a content-equivalence group never crosses a split boundary;
    - every pose contributes at least one group to each of train/val/test
      (requires >= 3 groups per pose);
    - splits are disjoint and exhaustive over the entries;
    - the same entries + seed always produce the same result.
    """
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

    short = {
        pose: len(keys) for pose, keys in pose_groups.items() if len(keys) < len(SPLIT_NAMES)
    }
    if short:
        raise ValueError(
            "cannot stratify splits by pose: pose(s) "
            + ", ".join(f"{pose!r} ({count} independent groups)" for pose, count in short.items())
            + f" need at least {len(SPLIT_NAMES)} content-equivalence groups each "
            "(one per split); collect more independent episodes for these poses "
            "or drop them from the dataset"
        )

    rng = np.random.default_rng(seed)
    assigned: dict[str, str] = {}
    counts = {name: 0 for name in SPLIT_NAMES}

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
        for name in SPLIT_NAMES
    }

    per_pose: dict[str, Any] = {}
    for pose in sorted(pose_groups):
        pose_counts = {name: 0 for name in SPLIT_NAMES}
        for key in pose_groups[pose]:
            pose_counts[assigned[key]] += len(groups[key])
        per_pose[pose] = {
            "n_groups": len(pose_groups[pose]),
            "n_episodes": sum(len(groups[key]) for key in pose_groups[pose]),
            "episodes_per_split": pose_counts,
        }
    metadata = {
        "strategy": "grouped_pose_stratified",
        "grouping": GROUPING_STRATEGY,
        "seed": seed,
        "fractions": list(fractions),
        "n_episodes": n,
        "n_groups": len(groups),
        "requested_sizes": {
            "train": n - n_val_target - n_test_target,
            "val": n_val_target,
            "test": n_test_target,
        },
        "actual_sizes": {name: counts[name] for name in SPLIT_NAMES},
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


def make_splits(
    episode_ids: list[str],
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Id-only splits (every episode its own group, single pose stratum).

    Only valid when every episode is known to be content-distinct — official
    dataset splits must go through :func:`make_grouped_splits` with entries
    from :func:`split_entries`, which enforces content grouping.
    """
    _check_fractions(fractions)
    entries = [
        SplitEntry(episode_id=str(episode_id), group_key=f"id:{episode_id}")
        for episode_id in episode_ids
    ]
    splits, _ = make_grouped_splits(entries, fractions=fractions, seed=seed)
    return splits


def _check_fractions(fractions: tuple[float, float, float]) -> None:
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError(f"fractions must be 3 positive values, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")


def split_fingerprint(splits: dict[str, list[str]]) -> str:
    """Order-independent identity of one split assignment (sha256)."""
    canonical = {name: sorted(splits[name]) for name in SPLIT_NAMES}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_no_cross_split_duplicates(
    entries: list[SplitEntry], splits: dict[str, list[str]]
) -> None:
    """Fail if any content-equivalence group spans more than one split."""
    membership = {eid: name for name in SPLIT_NAMES for eid in splits[name]}
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
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fractions": list(fractions),
        "seed": seed,
        "n_episodes": sum(len(ids) for ids in splits.values()),
        "split_fingerprint_sha256": split_fingerprint(splits),
        "splits": {name: splits[name] for name in SPLIT_NAMES},
        "created_utc": datetime.now(UTC).isoformat(),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_splits(
    path: str | Path, episode_ids: list[str] | None = None
) -> dict[str, list[str]]:
    """Load a split file; if ``episode_ids`` is given, reject a stale file."""
    payload = load_split_payload(path)
    splits = {name: list(payload["splits"][name]) for name in SPLIT_NAMES}

    all_ids = [eid for ids in splits.values() for eid in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f"split file {path} has overlapping splits")
    if episode_ids is not None and set(all_ids) != set(episode_ids):
        raise ValueError(
            f"split file {path} does not match the dataset episodes — the dataset "
            "was re-exported; regenerate the splits"
        )
    return splits


def load_split_payload(path: str | Path) -> dict[str, Any]:
    """Raw split-file payload (splits + fingerprint + grouping metadata)."""
    return json.loads(Path(path).read_text())


__all__ = [
    "DEFAULT_FRACTIONS",
    "DEFAULT_POSE_ID",
    "GROUPING_STRATEGY",
    "SPLIT_NAMES",
    "SplitEntry",
    "assert_no_cross_split_duplicates",
    "episode_content_key",
    "load_split_payload",
    "load_splits",
    "make_grouped_splits",
    "make_splits",
    "save_splits",
    "split_entries",
    "split_fingerprint",
    "splits_path",
]
