"""Dataset plumbing shared by chunk-policy trainers: splits, stats, factories.

Pure numpy on top of the frozen Phase 3.0 dataset interface (torch-free, so
gates and training scripts share one code path that can run before any
torch/Isaac initialization). Episodes are consumed only through
``EpisodeDataset`` / ``ChunkSampler`` / ``BatchIterator``. The dataset config
is duck-typed: anything with ``task`` / ``space`` / ``version`` /
``obs_preset`` fields works (ACT and Diffusion cfgs both do).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import AlexV2ContractError, RobotAssetRef
from alexdoor_xas.dataset import (
    BatchIterator,
    ChunkSampler,
    DatasetNormStats,
    EpisodeDataset,
    compute_norm_stats,
    load_norm_stats,
    load_splits,
    norm_stats_path,
    splits_path,
    validate_norm_stats,
)
from alexdoor_xas.dataset.robot_asset import (
    load_dataset_robot_asset,
    validate_dataset_episode_robot_asset,
)

EPOCH_SEED_STRIDE = 10_000
"""Per-epoch shuffle seed = ``train_seed * stride + epoch`` (fresh order every
epoch, reproducible across runs)."""

BatchNormalizer = Callable[[dict[str, Any], DatasetNormStats], dict[str, Any]]
"""Maps a raw ``BatchIterator`` batch to a normalized one. ACT uses the
default z-score ``normalize_batch``; Diffusion swaps in min-max actions."""


class PolicyDataError(ValueError):
    """Raised when the dataset/splits/stats triple is missing or stale."""


@dataclass(frozen=True)
class PolicyData:
    """One dataset export plus its shared splits and normalization stats."""

    dataset: EpisodeDataset
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    stats: DatasetNormStats
    stats_source: str  # "official" (norm_stats.json) | "computed" (non-default preset)
    robot_asset: RobotAssetRef | None
    robot_asset_manifest: dict[str, Any] | None

    @property
    def obs_dim(self) -> int:
        return self.stats.obs.dim

    @property
    def action_dim(self) -> int:
        return self.stats.action.dim


def load_policy_data(cfg, datasets_root: str | Path = paths.DATASETS_DIR) -> PolicyData:
    """Load the export named by ``cfg`` with hard staleness checks.

    Splits are the shared per-task file (rejects a stale pass via episode-id
    comparison). Norm stats come from the official ``norm_stats.json`` when its
    preset matches; for a non-default preset the obs stats are recomputed
    in-memory over the same train split (the official file only covers the
    default preset).
    """
    dataset_dir = Path(datasets_root) / cfg.task / cfg.space / cfg.version
    try:
        dataset = EpisodeDataset(dataset_dir)
    except FileNotFoundError as error:
        raise PolicyDataError(str(error)) from error

    try:
        robot_asset, robot_asset_manifest = load_dataset_robot_asset(
            dataset_dir, require=cfg.task == paths.ALEX_V2_TASK
        )
        if cfg.task == paths.ALEX_V2_TASK and robot_asset is not None:
            validate_dataset_episode_robot_asset(dataset, robot_asset)
    except AlexV2ContractError as error:
        raise PolicyDataError(f"invalid robot asset provenance: {error}") from error

    split_file = splits_path(datasets_root, cfg.task, cfg.version)
    if not split_file.is_file():
        raise PolicyDataError(
            f"splits file missing: {split_file} "
            "(run scripts/verify_dataset_interface.py --write-artifacts)"
        )
    try:
        splits = load_splits(split_file, episode_ids=dataset.episode_ids)
    except ValueError as error:
        raise PolicyDataError(f"stale or invalid splits file {split_file}: {error}") from error
    train_ids = list(splits["train"])

    stats_file = norm_stats_path(dataset_dir)
    if not stats_file.is_file():
        raise PolicyDataError(
            f"norm stats missing: {stats_file} "
            "(run scripts/verify_dataset_interface.py --write-artifacts)"
        )
    official = load_norm_stats(stats_file)
    if official.obs_preset == cfg.obs_preset:
        errors = validate_norm_stats(official, dataset, train_ids, obs_preset=cfg.obs_preset)
        if errors:
            raise PolicyDataError(
                f"norm stats {stats_file} do not match the dataset: " + "; ".join(errors)
            )
        stats, stats_source = official, "official"
    else:
        # Same train split, same code path as the official file — only the obs
        # preset differs, so the recomputed stats are equally deterministic.
        stats = compute_norm_stats(dataset, train_ids, obs_preset=cfg.obs_preset)
        stats_source = "computed"

    return PolicyData(
        dataset=dataset,
        train_ids=tuple(train_ids),
        val_ids=tuple(splits["val"]),
        test_ids=tuple(splits["test"]),
        stats=stats,
        stats_source=stats_source,
        robot_asset=robot_asset,
        robot_asset_manifest=robot_asset_manifest,
    )


def normalize_batch(batch: dict[str, Any], stats: DatasetNormStats) -> dict[str, Any]:
    """Z-score a ``BatchIterator`` batch's obs/actions (other keys pass through)."""
    normalized = dict(batch)
    normalized["obs"] = stats.obs.normalize(batch["obs"])
    normalized["actions"] = stats.action.normalize(batch["actions"])
    return normalized


def make_train_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...] | None = None,
    normalize: BatchNormalizer = normalize_batch,
):
    """Per-epoch reshuffled, normalized train batches (``TrainBatchFactory``)."""
    ids = list(episode_ids if episode_ids is not None else data.train_ids)
    sampler = ChunkSampler(
        data.dataset, chunk_size, obs_preset=data.stats.obs_preset, episode_ids=ids
    )
    drop_last = len(sampler) >= batch_size  # keep tiny overfit subsets trainable

    def factory(epoch: int):
        iterator = BatchIterator(
            sampler,
            batch_size,
            seed=seed * EPOCH_SEED_STRIDE + epoch,
            drop_last=drop_last,
        )
        return (normalize(batch, data.stats) for batch in iterator)

    return factory


def make_eval_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...],
    normalize: BatchNormalizer = normalize_batch,
):
    """Fixed-order normalized batches over a split (``ValBatchFactory``)."""
    sampler = ChunkSampler(
        data.dataset, chunk_size, obs_preset=data.stats.obs_preset, episode_ids=list(episode_ids)
    )

    def factory():
        iterator = BatchIterator(sampler, batch_size, seed=seed, drop_last=False)
        return (normalize(batch, data.stats) for batch in iterator)

    return factory


__all__ = [
    "EPOCH_SEED_STRIDE",
    "BatchNormalizer",
    "PolicyData",
    "PolicyDataError",
    "load_policy_data",
    "make_eval_factory",
    "make_train_factory",
    "normalize_batch",
]
