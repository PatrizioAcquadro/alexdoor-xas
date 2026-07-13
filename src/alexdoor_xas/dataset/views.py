"""Deterministic fixed-holdout, nested training views over one master dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .splits import SPLIT_NAMES, SplitEntry

VIEW_SCHEMA = "alexdoor_xas.dataset_view.v1"
SELECTION_ALGORITHM = "sha256(seed:pose_id:content_group):ascending"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def build_nested_views(
    entries: Sequence[SplitEntry],
    *,
    view_train_counts: Mapping[str, int],
    pose_ids: Sequence[str],
    seed: int,
    master_version: str,
    master_fingerprint: str,
    holdout_per_pose: int = 5,
) -> dict[str, dict[str, Any]]:
    """Build balanced fixed holdouts and per-pose nested training prefixes.

    Official scale masters are required to contain exactly 110 content-distinct
    episodes per pose. The first five deterministic content groups become
    validation, the next five become test, and the remaining 100 form nested
    training prefixes.
    """
    poses = tuple(str(pose) for pose in pose_ids)
    if len(poses) != 5 or len(set(poses)) != len(poses):
        raise ValueError("pose_ids must contain exactly five unique poses")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("selection seed must be a non-negative integer")
    if not master_version:
        raise ValueError("master_version must be non-empty")
    if SHA256_RE.fullmatch(master_fingerprint) is None:
        raise ValueError("master_fingerprint must be a lowercase SHA-256")
    if holdout_per_pose != 5:
        raise ValueError("each validation and test holdout must contain five episodes per pose")

    normalized_counts = _validated_view_counts(view_train_counts, len(poses))
    rows = list(entries)
    episode_ids = [entry.episode_id for entry in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("master episode ids are not unique")
    group_counts = Counter(entry.group_key for entry in rows)
    duplicates = sorted(key for key, count in group_counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "master episodes are not content-distinct; duplicate content group(s): "
            + ", ".join(key[:12] for key in duplicates[:8])
        )
    unknown_poses = sorted({entry.pose_id for entry in rows} - set(poses))
    if unknown_poses:
        raise ValueError(f"master contains unknown poses: {unknown_poses}")

    by_pose: dict[str, list[SplitEntry]] = {pose: [] for pose in poses}
    for entry in rows:
        by_pose[entry.pose_id].append(entry)
    per_pose_expected = 110
    wrong = {pose: len(items) for pose, items in by_pose.items() if len(items) != per_pose_expected}
    if wrong:
        raise ValueError(
            f"master must contain exactly {per_pose_expected} episodes per pose, got {wrong}"
        )
    if len(rows) != per_pose_expected * len(poses):
        raise ValueError(
            f"master must contain exactly {per_pose_expected * len(poses)} episodes"
        )

    ordered: dict[str, list[SplitEntry]] = {}
    for pose, items in by_pose.items():
        ordered[pose] = sorted(
            items,
            key=lambda entry: (
                hashlib.sha256(
                    f"{seed}:{pose}:{entry.group_key}".encode()
                ).hexdigest(),
                entry.episode_id,
            ),
        )

    val_ids = sorted(
        item.episode_id
        for pose in poses
        for item in ordered[pose][:holdout_per_pose]
    )
    test_ids = sorted(
        item.episode_id
        for pose in poses
        for item in ordered[pose][holdout_per_pose : 2 * holdout_per_pose]
    )
    train_pool = {
        pose: ordered[pose][2 * holdout_per_pose :] for pose in poses
    }

    views: dict[str, dict[str, Any]] = {}
    for view_id, n_train in normalized_counts.items():
        per_pose_train = n_train // len(poses)
        train_ids = sorted(
            item.episode_id
            for pose in poses
            for item in train_pool[pose][:per_pose_train]
        )
        payload: dict[str, Any] = {
            "schema": VIEW_SCHEMA,
            "view_id": view_id,
            "master_version": master_version,
            "master_dataset_fingerprint_sha256": master_fingerprint,
            "selection": {
                "algorithm": SELECTION_ALGORITHM,
                "seed": seed,
                "holdout_per_pose_per_split": holdout_per_pose,
                "holdouts_selected_once_before_training_prefixes": True,
            },
            "counts": {
                "train": n_train,
                "val": holdout_per_pose * len(poses),
                "test": holdout_per_pose * len(poses),
                "referenced": n_train + 2 * holdout_per_pose * len(poses),
            },
            "per_pose_counts": {
                pose: {
                    "train": per_pose_train,
                    "val": holdout_per_pose,
                    "test": holdout_per_pose,
                }
                for pose in poses
            },
            "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
            "content_grouping": "content_sha256",
        }
        payload["view_fingerprint_sha256"] = view_fingerprint(payload)
        views[view_id] = payload

    failures = validate_nested_views(
        views,
        rows,
        view_train_counts=normalized_counts,
        pose_ids=poses,
        master_version=master_version,
        master_fingerprint=master_fingerprint,
        holdout_per_pose=holdout_per_pose,
    )
    if failures:
        raise ValueError("generated nested views failed validation: " + "; ".join(failures))
    return views


def validate_nested_views(
    views: Mapping[str, Mapping[str, Any]],
    entries: Sequence[SplitEntry],
    *,
    view_train_counts: Mapping[str, int],
    pose_ids: Sequence[str],
    master_version: str,
    master_fingerprint: str,
    holdout_per_pose: int = 5,
) -> list[str]:
    """Return every balance, disjointness, grouping, nesting, or hash failure."""
    failures: list[str] = []
    poses = tuple(str(pose) for pose in pose_ids)
    expected_counts = dict(view_train_counts)
    if set(views) != set(expected_counts):
        failures.append(
            f"view inventory mismatch: expected={sorted(expected_counts)}, actual={sorted(views)}"
        )
        return failures
    entry_by_id = {entry.episode_id: entry for entry in entries}
    if len(entry_by_id) != len(entries):
        failures.append("master episode ids are duplicated")
        return failures
    group_counts = Counter(entry.group_key for entry in entries)
    if any(count != 1 for count in group_counts.values()):
        failures.append("master contains duplicate trajectory-content groups")

    fixed_val: list[str] | None = None
    fixed_test: list[str] | None = None
    previous_train: set[str] | None = None
    for view_id, n_train in expected_counts.items():
        payload = views[view_id]
        if payload.get("schema") != VIEW_SCHEMA:
            failures.append(f"{view_id}: schema mismatch")
        if payload.get("view_id") != view_id:
            failures.append(f"{view_id}: embedded view_id mismatch")
        if payload.get("master_version") != master_version:
            failures.append(f"{view_id}: master version mismatch")
        if payload.get("master_dataset_fingerprint_sha256") != master_fingerprint:
            failures.append(f"{view_id}: master fingerprint mismatch")
        try:
            splits = {
                name: list(payload["splits"][name]) for name in SPLIT_NAMES
            }
        except (KeyError, TypeError):
            failures.append(f"{view_id}: split mapping is missing or invalid")
            continue
        sizes = {name: len(ids) for name, ids in splits.items()}
        expected_sizes = {
            "train": n_train,
            "val": holdout_per_pose * len(poses),
            "test": holdout_per_pose * len(poses),
        }
        if sizes != expected_sizes:
            failures.append(f"{view_id}: split counts {sizes} != {expected_sizes}")
        all_ids = [episode_id for name in SPLIT_NAMES for episode_id in splits[name]]
        if len(all_ids) != len(set(all_ids)):
            failures.append(f"{view_id}: train/val/test overlap")
        missing = sorted(set(all_ids) - set(entry_by_id))
        if missing:
            failures.append(f"{view_id}: split IDs are absent from the master: {missing[:4]}")
        memberships = {
            episode_id: name for name in SPLIT_NAMES for episode_id in splits[name]
        }
        group_memberships: dict[str, set[str]] = {}
        for episode_id, split_name in memberships.items():
            if episode_id in entry_by_id:
                group_memberships.setdefault(entry_by_id[episode_id].group_key, set()).add(
                    split_name
                )
        if any(len(names) != 1 for names in group_memberships.values()):
            failures.append(f"{view_id}: trajectory-content group crosses a split")
        for split_name, expected_per_pose in (
            ("train", n_train // len(poses)),
            ("val", holdout_per_pose),
            ("test", holdout_per_pose),
        ):
            counts = Counter(
                entry_by_id[episode_id].pose_id
                for episode_id in splits[split_name]
                if episode_id in entry_by_id
            )
            if counts != Counter({pose: expected_per_pose for pose in poses}):
                failures.append(
                    f"{view_id}: {split_name} pose balance mismatch: {dict(counts)}"
                )
        if fixed_val is None:
            fixed_val, fixed_test = splits["val"], splits["test"]
        elif splits["val"] != fixed_val or splits["test"] != fixed_test:
            failures.append(f"{view_id}: fixed holdout IDs changed")
        current_train = set(splits["train"])
        if previous_train is not None and not previous_train < current_train:
            failures.append(f"{view_id}: training IDs are not a strict nested superset")
        previous_train = current_train
        if payload.get("view_fingerprint_sha256") != view_fingerprint(payload):
            failures.append(f"{view_id}: view fingerprint is stale")
    return failures


def view_fingerprint(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of a view payload, excluding its embedded digest."""
    canonical = {
        key: value for key, value in payload.items() if key != "view_fingerprint_sha256"
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_view_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish one already-validated deterministic view payload."""
    target = Path(path)
    if payload.get("view_fingerprint_sha256") != view_fingerprint(payload):
        raise ValueError("view fingerprint is stale")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_view_payload(path: str | Path) -> dict[str, Any]:
    """Load one view and reject schema or fingerprint tampering."""
    target = Path(path)
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != VIEW_SCHEMA:
        raise ValueError(f"invalid dataset view schema: {target}")
    if payload.get("view_fingerprint_sha256") != view_fingerprint(payload):
        raise ValueError(f"dataset view fingerprint mismatch: {target}")
    return payload


def view_path(datasets_root: str | Path, task: str, view_id: str) -> Path:
    return Path(datasets_root) / task / "splits" / f"{view_id}.json"


def _validated_view_counts(
    view_train_counts: Mapping[str, int], pose_count: int
) -> dict[str, int]:
    counts = dict(view_train_counts)
    expected = {
        "v3_scale_n50": 50,
        "v3_scale_n100": 100,
        "v3_scale_n250": 250,
        "v3_scale_n500": 500,
    }
    if counts != expected:
        raise ValueError(f"view training counts must be exactly {expected}")
    if any(value % pose_count for value in counts.values()):
        raise ValueError("view training counts must be evenly divisible across poses")
    return counts


__all__ = [
    "SELECTION_ALGORITHM",
    "VIEW_SCHEMA",
    "build_nested_views",
    "load_view_payload",
    "save_view_payload",
    "validate_nested_views",
    "view_fingerprint",
    "view_path",
]
