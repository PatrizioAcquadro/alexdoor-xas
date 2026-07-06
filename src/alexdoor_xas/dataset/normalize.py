"""Action/observation normalization statistics (Phase 3.0).

Per-dimension mean/std (plus min/max) computed **over the train split only**,
saved as ``norm_stats.json`` inside the dataset version directory — a stats
file describes exactly one generation pass and dies with it on re-export.
``std`` is floored so constant dimensions stay finite: the A2/A3 rotation
deltas are recorded but never actuated (frozen action contract), so their std
is exactly zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loader import DEFAULT_OBS_PRESET, EpisodeDataset, obs_matrix

STD_FLOOR = 1e-8
NORM_STATS_FILENAME = "norm_stats.json"


@dataclass(frozen=True)
class NormStats:
    """Per-dimension statistics of one quantity (actions or an obs preset)."""

    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,), floored at STD_FLOOR
    min: np.ndarray  # (D,)
    max: np.ndarray  # (D,)
    count: int  # total rows aggregated

    @classmethod
    def from_rows(cls, arrays: list[np.ndarray]) -> NormStats:
        """Aggregate a list of ``(N_i, D)`` arrays into one stats record."""
        if not arrays:
            raise ValueError("cannot compute stats from an empty list")
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays])
        if stacked.ndim != 2 or stacked.shape[0] == 0:
            raise ValueError(f"expected non-empty (N, D) rows, got shape {stacked.shape}")
        return cls(
            mean=stacked.mean(axis=0),
            std=np.maximum(stacked.std(axis=0), STD_FLOOR),
            min=stacked.min(axis=0),
            max=stacked.max(axis=0),
            count=int(stacked.shape[0]),
        )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormStats:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            min=np.asarray(data["min"], dtype=np.float64),
            max=np.asarray(data["max"], dtype=np.float64),
            count=int(data["count"]),
        )


@dataclass(frozen=True)
class DatasetNormStats:
    """Train-split stats of one dataset: actions + one obs preset."""

    action: NormStats
    obs: NormStats
    obs_preset: str
    train_episode_ids: tuple[str, ...]


def compute_norm_stats(
    dataset: EpisodeDataset,
    train_episode_ids: list[str],
    obs_preset: str = DEFAULT_OBS_PRESET,
) -> DatasetNormStats:
    """Compute action + observation stats over the train-split episodes."""
    records = [dataset.by_id(episode_id) for episode_id in train_episode_ids]
    if not records:
        raise ValueError("train split is empty")
    return DatasetNormStats(
        action=NormStats.from_rows([record.actions for record in records]),
        obs=NormStats.from_rows([obs_matrix(record, obs_preset) for record in records]),
        obs_preset=obs_preset,
        train_episode_ids=tuple(train_episode_ids),
    )


def norm_stats_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / NORM_STATS_FILENAME


def save_norm_stats(path: str | Path, stats: DatasetNormStats) -> Path:
    path = Path(path)
    payload = {
        "action": stats.action.to_dict(),
        "obs": stats.obs.to_dict(),
        "obs_preset": stats.obs_preset,
        "train_episode_ids": list(stats.train_episode_ids),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_norm_stats(path: str | Path) -> DatasetNormStats:
    payload = json.loads(Path(path).read_text())
    return DatasetNormStats(
        action=NormStats.from_dict(payload["action"]),
        obs=NormStats.from_dict(payload["obs"]),
        obs_preset=str(payload["obs_preset"]),
        train_episode_ids=tuple(payload["train_episode_ids"]),
    )


__all__ = [
    "NORM_STATS_FILENAME",
    "STD_FLOOR",
    "DatasetNormStats",
    "NormStats",
    "compute_norm_stats",
    "load_norm_stats",
    "norm_stats_path",
    "save_norm_stats",
]
