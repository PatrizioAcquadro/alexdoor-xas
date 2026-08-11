"""Self-contained ACT checkpoint v2 with legacy Phase 3 loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset import DatasetNormStats
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.common.checkpoint import (
    dataset_descriptor,
    robot_asset_from_payload,
    stats_from_payload,
    stats_payload,
    validate_checkpoint_contract,
)

CHECKPOINT_FORMAT = "alexdoor_xas.act.v2"
LEGACY_CHECKPOINT_FORMAT = "alexdoor_xas.act.v1"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: ACTModel
    config: dict[str, Any]
    stats: DatasetNormStats
    meta: dict[str, Any]
    checkpoint_format: str
    robot_asset: RobotAssetRef | None = None

    @property
    def action_space(self) -> str:
        return str(self.config["dataset"]["space"])

    @property
    def obs_preset(self) -> str:
        return str(self.config["dataset"]["obs_preset"])

    @property
    def chunk_size(self) -> int:
        return self.model.cfg.chunk_size


def save_checkpoint(
    path: str | Path,
    model: ACTModel,
    config: dict[str, Any],
    stats: DatasetNormStats,
    meta: dict[str, Any] | None = None,
    robot_asset: RobotAssetRef | None = None,
) -> Path:
    """Write the compact v2 checkpoint and return its path."""

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
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": state_dict,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "model_cfg": asdict(model.cfg),
            "dataset": dataset,
            "norm_stats": stats_payload(stats),
            "robot_asset": robot_asset.to_dict() if robot_asset is not None else None,
            "meta": dict(meta or {}),
        },
        target,
    )
    return target


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """Load v2 or a legacy v1 checkpoint without consulting the live dataset."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    checkpoint_format = payload.get("format")
    if checkpoint_format not in {CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT}:
        raise ValueError(f"unsupported checkpoint format {checkpoint_format!r} in {path}")
    try:
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        model_cfg = ActModelCfg(**payload["model_cfg"])
        state_dict = payload["state_dict"]
        source_config = (
            payload.get("config") if checkpoint_format == LEGACY_CHECKPOINT_FORMAT else None
        )
        dataset = dataset_descriptor(
            source_config if isinstance(source_config, dict) else {"dataset": payload["dataset"]}
        )
        stats = stats_from_payload(payload["norm_stats"])
        robot_asset = robot_asset_from_payload(payload.get("robot_asset"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid ACT checkpoint {path}: {error}") from error
    validate_checkpoint_contract(
        dataset=dataset,
        stats=stats,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dict=state_dict,
        robot_asset=robot_asset,
    )
    model = ACTModel(obs_dim=obs_dim, action_dim=action_dim, cfg=model_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a mapping")
    return LoadedCheckpoint(
        model=model,
        config={"dataset": dataset},
        stats=stats,
        meta=dict(meta),
        checkpoint_format=str(checkpoint_format),
        robot_asset=robot_asset,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "LEGACY_CHECKPOINT_FORMAT",
    "LoadedCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
]
