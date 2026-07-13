"""Action/observation normalization statistics (Phase 3.0).

Per-dimension mean/std (plus min/max) computed **over the train split only**,
saved as ``norm_stats.json`` inside the dataset version directory — a stats
file describes exactly one generation pass and dies with it on re-export.
``std`` is floored so constant dimensions stay finite: the A2/A3 rotation
deltas are recorded but never actuated (frozen action contract), so their std
is exactly zero.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loader import DEFAULT_OBS_PRESET, EpisodeDataset, obs_matrix

STD_FLOOR = 1e-8
NORM_STATS_FILENAME = "norm_stats.json"
DATASET_FINGERPRINT_CONTRACT = "alexdoor_xas.dataset_fingerprint.v2"


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
        normalized = [np.asarray(array, dtype=np.float64) for array in arrays]
        if any(array.ndim != 2 or array.shape[0] == 0 for array in normalized):
            raise ValueError("normalization inputs must all be non-empty (N, D) arrays")
        dimensions = {array.shape[1] for array in normalized}
        if len(dimensions) != 1:
            raise ValueError("normalization inputs have inconsistent feature dimensions")
        if not all(np.isfinite(array).all() for array in normalized):
            raise ValueError("normalization inputs must be finite")
        stacked = np.concatenate(normalized)
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
    dataset_episode_ids: tuple[str, ...] = ()
    action_space: str = ""
    dataset_fingerprint: str = ""
    split_name: str = "train"
    view_id: str | None = None
    view_fingerprint: str = ""
    normalization_fingerprint: str = ""
    normalization_sha256: str = ""


def compute_norm_stats(
    dataset: EpisodeDataset,
    train_episode_ids: list[str],
    obs_preset: str = DEFAULT_OBS_PRESET,
    *,
    view_id: str | None = None,
    view_fingerprint: str = "",
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
        dataset_episode_ids=tuple(dataset.episode_ids),
        action_space=dataset.action_space,
        dataset_fingerprint=dataset_fingerprint(dataset, obs_preset),
        view_id=view_id,
        view_fingerprint=view_fingerprint,
    )


def norm_stats_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / NORM_STATS_FILENAME


def view_norm_stats_path(dataset_dir: str | Path, view_id: str) -> Path:
    """Canonical per-action-space normalization artifact for one dataset view."""
    if not isinstance(view_id, str) or not view_id or "/" in view_id or ".." in view_id:
        raise ValueError("view_id must be a safe single path component")
    return Path(dataset_dir) / "views" / view_id / NORM_STATS_FILENAME


def save_norm_stats(path: str | Path, stats: DatasetNormStats) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _stats_payload(stats)
    payload["normalization_fingerprint_sha256"] = normalization_fingerprint(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_norm_stats(path: str | Path) -> DatasetNormStats:
    source = Path(path)
    payload = json.loads(source.read_text())
    return DatasetNormStats(
        action=NormStats.from_dict(payload["action"]),
        obs=NormStats.from_dict(payload["obs"]),
        obs_preset=str(payload["obs_preset"]),
        train_episode_ids=tuple(payload["train_episode_ids"]),
        dataset_episode_ids=tuple(payload.get("dataset_episode_ids", ())),
        action_space=str(payload.get("action_space", "")),
        dataset_fingerprint=str(payload.get("dataset_fingerprint", "")),
        split_name=str(payload.get("split_name", "train")),
        view_id=str(payload["view_id"]) if payload.get("view_id") is not None else None,
        view_fingerprint=str(payload.get("view_fingerprint_sha256", "")),
        normalization_fingerprint=str(
            payload.get("normalization_fingerprint_sha256", "")
        ),
        normalization_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def normalization_fingerprint(stats_or_payload: DatasetNormStats | dict[str, Any]) -> str:
    """Canonical hash of normalization content, excluding its embedded digest."""
    payload = (
        _stats_payload(stats_or_payload)
        if isinstance(stats_or_payload, DatasetNormStats)
        else dict(stats_or_payload)
    )
    payload.pop("normalization_fingerprint_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dataset_fingerprint(dataset: EpisodeDataset, obs_preset: str = DEFAULT_OBS_PRESET) -> str:
    """Preset-specific content fingerprint for stat/checkpoint compatibility."""
    digest = hashlib.sha256()
    digest.update(DATASET_FINGERPRINT_CONTRACT.encode())
    digest.update(b"\0obs_preset\0")
    digest.update(obs_preset.encode())
    digest.update(b"\0")
    digest.update(dataset.action_space.encode())
    digest.update(dataset.task.encode())
    robot_asset = dataset.meta.get("robot_asset")
    has_robot_provenance = robot_asset is not None or any(
        record.meta.get("robot_asset_id") or record.meta.get("robot_asset_sha256")
        for record in dataset.records
    )
    # Preserve the exact Phase 3.0/V1 digest byte stream when provenance is
    # absent.  V2 adds a domain-separated canonical payload and episode refs.
    if has_robot_provenance:
        digest.update(b"\0robot_asset\0")
        digest.update(
            json.dumps(robot_asset, sort_keys=True, separators=(",", ":")).encode()
        )
    for record in sorted(dataset.records, key=lambda r: r.episode_id):
        digest.update(record.episode_id.encode())
        for key in ("seed", "robot", "scene", "policy"):
            digest.update(str(record.meta.get(key, "")).encode())
            digest.update(b"\0")
        if has_robot_provenance:
            for key in ("robot_asset_id", "robot_asset_sha256"):
                digest.update(str(record.meta.get(key, "")).encode())
                digest.update(b"\0")
        digest.update(str(record.success).encode())
        digest.update(np.asarray([record.final_door_angle], dtype=np.float64).tobytes())
        digest.update(str(record.failure_label).encode())
        digest.update(np.asarray(record.t, dtype=np.float64).tobytes())
        digest.update(np.asarray(record.actions, dtype=np.float64).tobytes())
        digest.update(obs_matrix(record, obs_preset).astype(np.float64).tobytes())
    return digest.hexdigest()


def validate_norm_stats(
    stats: DatasetNormStats,
    dataset: EpisodeDataset,
    train_episode_ids: list[str],
    obs_preset: str = DEFAULT_OBS_PRESET,
    split_name: str = "train",
    view_id: str | None = None,
    view_fingerprint: str = "",
) -> list[str]:
    """Return compatibility errors for loaded normalization stats."""
    errors: list[str] = []
    expected_ids = tuple(dataset.episode_ids)
    expected_train = tuple(train_episode_ids)
    if stats.action_space != dataset.action_space:
        errors.append(
            f"norm stats action_space {stats.action_space!r} != dataset {dataset.action_space!r}"
        )
    if not stats.dataset_episode_ids:
        errors.append(
            "norm stats carry no dataset_episode_ids provenance; regenerate them "
            "with the current fingerprint contract"
        )
    elif stats.dataset_episode_ids != expected_ids:
        errors.append("norm stats dataset_episode_ids do not match the dataset")
    if stats.train_episode_ids != expected_train:
        errors.append("norm stats train_episode_ids do not match the requested train split")
    if stats.split_name != split_name:
        errors.append(f"norm stats split_name {stats.split_name!r} != {split_name!r}")
    if stats.view_id != view_id:
        errors.append(f"norm stats view_id {stats.view_id!r} != {view_id!r}")
    if view_id is not None:
        if not view_fingerprint:
            errors.append("requested dataset view has no view fingerprint")
        if stats.view_fingerprint != view_fingerprint:
            errors.append("norm stats view fingerprint does not match the requested view")
        if not stats.normalization_fingerprint:
            errors.append("view norm stats carry no normalization fingerprint")
        elif stats.normalization_fingerprint != normalization_fingerprint(stats):
            errors.append("view norm stats normalization fingerprint is stale")
    if stats.obs_preset != obs_preset:
        errors.append(f"norm stats obs_preset {stats.obs_preset!r} != {obs_preset!r}")
    expected_fingerprint = dataset_fingerprint(dataset, obs_preset)
    if stats.dataset_fingerprint != expected_fingerprint:
        errors.append("norm stats dataset_fingerprint does not match the dataset content")
    if stats.action.dim != dataset.action_dim:
        errors.append(f"norm stats action dim {stats.action.dim} != dataset {dataset.action_dim}")
    expected_obs_dim = obs_matrix(dataset[0], obs_preset).shape[1]
    if stats.obs.dim != expected_obs_dim:
        errors.append(f"norm stats obs dim {stats.obs.dim} != preset {expected_obs_dim}")
    for name, block in (("action", stats.action), ("obs", stats.obs)):
        dims = {block.mean.shape, block.std.shape, block.min.shape, block.max.shape}
        if len(dims) != 1 or block.mean.ndim != 1:
            errors.append(f"norm stats {name} arrays must be 1-D with matching shapes")
        if not (
            np.isfinite(block.mean).all()
            and np.isfinite(block.std).all()
            and np.isfinite(block.min).all()
            and np.isfinite(block.max).all()
        ):
            errors.append(f"norm stats {name} arrays must be finite")
        if (block.std < STD_FLOOR).any():
            errors.append(f"norm stats {name} std is below STD_FLOOR")
        if block.count <= 0:
            errors.append(f"norm stats {name} count must be positive")
    try:
        recomputed = compute_norm_stats(
            dataset,
            train_episode_ids,
            obs_preset,
            view_id=view_id,
            view_fingerprint=view_fingerprint,
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        errors.append(f"normalization numerical recomputation failed: {error}")
    else:
        for name in ("action", "obs"):
            stored_block = getattr(stats, name)
            expected_block = getattr(recomputed, name)
            if stored_block.count != expected_block.count:
                errors.append(
                    f"norm stats recomputed {name} count mismatch: "
                    f"stored={stored_block.count}, expected={expected_block.count}"
                )
            for field in ("mean", "std", "min", "max"):
                stored_value = getattr(stored_block, field)
                expected_value = getattr(expected_block, field)
                if not np.array_equal(stored_value, expected_value):
                    errors.append(f"norm stats recomputed {name} {field} mismatch")
    return errors


def _stats_payload(stats: DatasetNormStats) -> dict[str, Any]:
    return {
        "action": stats.action.to_dict(),
        "obs": stats.obs.to_dict(),
        "obs_preset": stats.obs_preset,
        "train_episode_ids": list(stats.train_episode_ids),
        "dataset_episode_ids": list(stats.dataset_episode_ids),
        "action_space": stats.action_space,
        "dataset_fingerprint": stats.dataset_fingerprint,
        "split_name": stats.split_name,
        "view_id": stats.view_id,
        "view_fingerprint_sha256": stats.view_fingerprint,
    }


__all__ = [
    "DATASET_FINGERPRINT_CONTRACT",
    "NORM_STATS_FILENAME",
    "STD_FLOOR",
    "DatasetNormStats",
    "NormStats",
    "compute_norm_stats",
    "dataset_fingerprint",
    "load_norm_stats",
    "normalization_fingerprint",
    "norm_stats_path",
    "save_norm_stats",
    "validate_norm_stats",
    "view_norm_stats_path",
]
