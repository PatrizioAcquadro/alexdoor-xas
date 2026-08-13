"""Dataset loading, validation, normalization, and batch factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import AlexV2ContractError, RobotAssetRef
from alexdoor_xas.dataset.loader import EpisodeDataset
from alexdoor_xas.dataset.normalize import (
    DatasetNormStats,
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path,
    validate_norm_stats,
    view_norm_stats_path,
)
from alexdoor_xas.dataset.robot_asset import (
    load_dataset_robot_asset,
    validate_dataset_episode_robot_asset,
)
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler
from alexdoor_xas.dataset.splits import load_splits, load_view_splits, splits_path, view_path

EPOCH_SEED_STRIDE = 10_000

BatchNormalizer = Callable[[dict[str, Any], DatasetNormStats], dict[str, Any]]


class PolicyDataError(ValueError):
    """Invalid dataset, split, or normalization contract."""


@dataclass(frozen=True)
class PolicyData:
    """Validated dataset inputs for policy training."""

    dataset: EpisodeDataset
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    stats: DatasetNormStats
    robot_asset: RobotAssetRef | None

    @property
    def obs_dim(self) -> int:
        return self.stats.obs.dim

    @property
    def action_dim(self) -> int:
        return self.stats.action.dim


def load_policy_data(cfg, datasets_root: str | Path = paths.DATASETS_DIR) -> PolicyData:
    """Load and validate dataset, splits, robot identity, and statistics."""
    dataset_dir = Path(datasets_root) / cfg.task / cfg.space / cfg.version
    try:
        dataset = EpisodeDataset(dataset_dir)
    except FileNotFoundError as error:
        raise PolicyDataError(str(error)) from error

    try:
        robot_asset, _ = load_dataset_robot_asset(
            dataset_dir, require=cfg.task == paths.ALEX_V2_TASK
        )
        if cfg.task == paths.ALEX_V2_TASK and robot_asset is not None:
            validate_dataset_episode_robot_asset(dataset, robot_asset)
    except AlexV2ContractError as error:
        raise PolicyDataError(f"invalid robot asset provenance: {error}") from error

    selected_view = getattr(cfg, "view_id", None)
    if selected_view is None:
        split_file = splits_path(datasets_root, cfg.task, cfg.version)
    else:
        split_file = view_path(datasets_root, cfg.task, selected_view)
    if not split_file.is_file():
        raise PolicyDataError(
            f"splits file missing: {split_file} "
            "(run scripts/verify_dataset_interface.py --write-artifacts)"
        )
    try:
        if selected_view is None:
            splits = load_splits(split_file, episode_ids=dataset.episode_ids)
        else:
            splits = load_view_splits(
                split_file,
                view_id=selected_view,
                master_version=cfg.version,
                episode_ids=dataset.episode_ids,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise PolicyDataError(f"stale or invalid splits file {split_file}: {error}") from error
    train_ids = list(splits["train"])

    stats_file = (
        norm_stats_path(dataset_dir)
        if selected_view is None
        else view_norm_stats_path(dataset_dir, selected_view)
    )
    if not stats_file.is_file():
        raise PolicyDataError(
            f"norm stats missing: {stats_file} "
            "(run scripts/verify_dataset_interface.py --write-artifacts)"
        )
    official = load_norm_stats(stats_file)
    official_errors = validate_norm_stats(
        official,
        dataset,
        train_ids,
        obs_preset=official.obs_preset,
        view_id=selected_view,
    )
    if official_errors:
        raise PolicyDataError(
            f"norm stats {stats_file} do not match the dataset: " + "; ".join(official_errors)
        )
    if official.obs_preset == cfg.obs_preset:
        stats = official
    elif selected_view is not None:
        raise PolicyDataError(
            f"view normalization {stats_file} uses {official.obs_preset!r}, not "
            f"requested preset {cfg.obs_preset!r}; view runs must use their committed "
            "train-only normalization artifact"
        )
    else:
        stats = compute_norm_stats(dataset, train_ids, obs_preset=cfg.obs_preset)

    return PolicyData(
        dataset=dataset,
        train_ids=tuple(train_ids),
        val_ids=tuple(splits["val"]),
        stats=stats,
        robot_asset=robot_asset,
    )


def normalize_batch(batch: dict[str, Any], stats: DatasetNormStats) -> dict[str, Any]:
    """Z-score observations and actions."""
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
    drop_last = len(sampler) >= batch_size

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
