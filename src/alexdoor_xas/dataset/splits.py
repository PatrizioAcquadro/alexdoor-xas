"""Deterministic train/val/test splits, shared across action spaces (Phase 3.0).

Episode ids are identical across every action-space export of one generation
pass (``data_engine/export.py`` relabels the same episodes in place), so splits
are computed **once per task/version** and shared by A1–A4 — matched-condition
action-space comparisons require it. The split file lives at
``datasets/<task>/splits/<version>.json``; a re-export mints fresh episode ids,
so re-exporting a version means regenerating its splits (same command, same
result for the same seed).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.75, 0.125, 0.125)


def make_splits(
    episode_ids: list[str],
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministic disjoint splits covering all ids (>=1 episode in val/test)."""
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError(f"fractions must be 3 positive values, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode ids are not unique")
    n = len(episode_ids)
    if n < 3:
        raise ValueError(f"need at least 3 episodes to split, got {n}")

    n_val = max(1, round(fractions[1] * n))
    n_test = max(1, round(fractions[2] * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"fractions {fractions} leave no training episodes for n={n}")

    permuted = np.random.default_rng(seed).permutation(sorted(episode_ids))
    return {
        "train": [str(e) for e in permuted[:n_train]],
        "val": [str(e) for e in permuted[n_train : n_train + n_val]],
        "test": [str(e) for e in permuted[n_train + n_val :]],
    }


def splits_path(datasets_root: str | Path, task: str, version: str) -> Path:
    """Canonical split-file location: per task + version, shared across spaces."""
    return Path(datasets_root) / task / "splits" / f"{version}.json"


def save_splits(
    path: str | Path,
    splits: dict[str, list[str]],
    *,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fractions": list(fractions),
        "seed": seed,
        "n_episodes": sum(len(ids) for ids in splits.values()),
        "splits": {name: splits[name] for name in SPLIT_NAMES},
        "created_utc": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_splits(
    path: str | Path, episode_ids: list[str] | None = None
) -> dict[str, list[str]]:
    """Load a split file; if ``episode_ids`` is given, reject a stale file."""
    payload = json.loads(Path(path).read_text())
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


__all__ = [
    "DEFAULT_FRACTIONS",
    "SPLIT_NAMES",
    "load_splits",
    "make_splits",
    "save_splits",
    "splits_path",
]
