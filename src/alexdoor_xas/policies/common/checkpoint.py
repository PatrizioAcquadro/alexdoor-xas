"""Shared validation for compact, self-contained learned-policy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset import DatasetNormStats, NormStats
from alexdoor_xas.policies.common.runs import torch_save_atomic

DATASET_FIELDS = ("task", "space", "version", "obs_preset", "view_id")


@dataclass(frozen=True)
class CheckpointPayload:
    """Validated model-neutral fields from one learned-policy checkpoint."""

    state_dict: Mapping[str, Any]
    obs_dim: int
    action_dim: int
    model_cfg: dict[str, Any]
    dataset: dict[str, Any]
    stats: DatasetNormStats
    meta: dict[str, Any]
    checkpoint_format: str
    robot_asset: RobotAssetRef | None


def dataset_descriptor(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config.get("dataset")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint config requires a dataset mapping")
    descriptor = {field: source.get(field) for field in DATASET_FIELDS}
    for field in ("task", "space", "version", "obs_preset"):
        if not isinstance(descriptor[field], str) or not descriptor[field]:
            raise ValueError(f"checkpoint dataset {field} must be a non-empty string")
    view_id = descriptor["view_id"]
    if view_id is not None and (not isinstance(view_id, str) or not view_id):
        raise ValueError("checkpoint dataset view_id must be null or a non-empty string")
    return descriptor


def stats_payload(stats: DatasetNormStats) -> dict[str, Any]:
    return {
        "action": stats.action.to_dict(),
        "obs": stats.obs.to_dict(),
        "obs_preset": stats.obs_preset,
        "train_episode_ids": list(stats.train_episode_ids),
        "action_space": stats.action_space,
        "view_id": stats.view_id,
    }


def stats_from_payload(payload: Mapping[str, Any]) -> DatasetNormStats:
    try:
        view_id = payload.get("view_id")
        return DatasetNormStats(
            action=NormStats.from_dict(dict(payload["action"])),
            obs=NormStats.from_dict(dict(payload["obs"])),
            obs_preset=str(payload["obs_preset"]),
            train_episode_ids=tuple(str(item) for item in payload["train_episode_ids"]),
            action_space=str(payload["action_space"]),
            view_id=str(view_id) if view_id is not None else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint normalization stats: {error}") from error


def validate_checkpoint_contract(
    *,
    dataset: Mapping[str, Any],
    stats: DatasetNormStats,
    obs_dim: int,
    action_dim: int,
    state_dict: Mapping[str, Any],
    robot_asset: RobotAssetRef | None,
) -> None:
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("checkpoint dimensions must be positive")
    if dataset.get("space") != stats.action_space:
        raise ValueError("checkpoint action space does not match normalization stats")
    if dataset.get("obs_preset") != stats.obs_preset:
        raise ValueError("checkpoint observation preset does not match normalization stats")
    if dataset.get("view_id") != stats.view_id:
        raise ValueError("checkpoint dataset view does not match normalization stats")
    if stats.obs.dim != obs_dim or stats.action.dim != action_dim:
        raise ValueError("checkpoint dimensions do not match normalization stats")
    if not stats.train_episode_ids:
        raise ValueError("checkpoint normalization train split is empty")
    for label, block in (("action", stats.action), ("obs", stats.obs)):
        shapes = {block.mean.shape, block.std.shape, block.min.shape, block.max.shape}
        if len(shapes) != 1 or block.mean.ndim != 1:
            raise ValueError(f"checkpoint {label} stats arrays must be matching vectors")
        if not all(
            np.isfinite(value).all() for value in (block.mean, block.std, block.min, block.max)
        ):
            raise ValueError(f"checkpoint {label} stats must be finite")
        if (block.std <= 0.0).any() or (block.min > block.max).any() or block.count <= 0:
            raise ValueError(f"checkpoint {label} stats are invalid")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint weight {name!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint weight {name!r} contains non-finite values")
    if dataset.get("task") == paths.ALEX_V2_TASK and robot_asset is None:
        raise ValueError("Alex V2 checkpoints require robot identity")


def robot_asset_from_payload(payload: Any) -> RobotAssetRef | None:
    if payload is None:
        return None
    try:
        asset = RobotAssetRef.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint robot identity: {error}") from error
    return asset


def save_checkpoint_payload(
    path: str | Path,
    checkpoint_format: str,
    model: Any,
    config: Mapping[str, Any],
    stats: DatasetNormStats,
    meta: Mapping[str, Any] | None = None,
    robot_asset: RobotAssetRef | None = None,
) -> Path:
    """Validate and atomically write the shared v2 checkpoint payload."""
    dataset = dataset_descriptor(config)
    state_dict = model.state_dict()
    validate_checkpoint_contract(
        dataset=dataset,
        stats=stats,
        obs_dim=model.obs_dim,
        action_dim=model.action_dim,
        state_dict=state_dict,
        robot_asset=robot_asset,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch_save_atomic(
        target,
        {
            "format": checkpoint_format,
            "state_dict": state_dict,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "model_cfg": asdict(model.cfg),
            "dataset": dataset,
            "norm_stats": stats_payload(stats),
            "robot_asset": robot_asset.to_dict() if robot_asset is not None else None,
            "meta": dict(meta or {}),
        },
    )
    return target


def load_checkpoint_payload(
    path: str | Path,
    expected_format: str,
    checkpoint_label: str,
    map_location: str = "cpu",
) -> CheckpointPayload:
    """Load and validate model-neutral v2 checkpoint fields."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    checkpoint_format = payload.get("format")
    if checkpoint_format != expected_format:
        raise ValueError(f"unsupported checkpoint format {checkpoint_format!r} in {path}")
    try:
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        raw_model_cfg = payload["model_cfg"]
        if not isinstance(raw_model_cfg, Mapping):
            raise TypeError("model_cfg must be a mapping")
        model_cfg = dict(raw_model_cfg)
        state_dict = payload["state_dict"]
        dataset = dataset_descriptor({"dataset": payload["dataset"]})
        stats = stats_from_payload(payload["norm_stats"])
        robot_asset = robot_asset_from_payload(payload.get("robot_asset"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {checkpoint_label} checkpoint {path}: {error}"
        ) from error
    validate_checkpoint_contract(
        dataset=dataset,
        stats=stats,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dict=state_dict,
        robot_asset=robot_asset,
    )
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a mapping")
    return CheckpointPayload(
        state_dict=state_dict,
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_cfg=model_cfg,
        dataset=dataset,
        stats=stats,
        meta=dict(meta),
        checkpoint_format=str(checkpoint_format),
        robot_asset=robot_asset,
    )


__all__ = [
    "CheckpointPayload",
    "dataset_descriptor",
    "load_checkpoint_payload",
    "robot_asset_from_payload",
    "save_checkpoint_payload",
    "stats_from_payload",
    "stats_payload",
    "validate_checkpoint_contract",
]
