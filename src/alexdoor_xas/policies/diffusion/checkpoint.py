"""Self-contained Diffusion checkpoint v2 loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset import DatasetNormStats
from alexdoor_xas.policies.common.checkpoint import (
    load_checkpoint_payload,
    save_checkpoint_payload,
)
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.diffusion.model import DiffusionTransformer

CHECKPOINT_FORMAT = "alexdoor_xas.diffusion.v2"


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: DiffusionTransformer
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
    def horizon(self) -> int:
        return self.model.cfg.horizon


def save_checkpoint(
    path: str | Path,
    model: DiffusionTransformer,
    config: dict[str, Any],
    stats: DatasetNormStats,
    meta: dict[str, Any] | None = None,
    robot_asset: RobotAssetRef | None = None,
) -> Path:
    """Write the compact v2 checkpoint and return its path."""
    return save_checkpoint_payload(
        path,
        CHECKPOINT_FORMAT,
        model,
        config,
        stats,
        meta,
        robot_asset,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """Load a v2 checkpoint without consulting the live dataset."""
    payload = load_checkpoint_payload(
        path, CHECKPOINT_FORMAT, "Diffusion", map_location
    )
    try:
        model_cfg = DiffusionModelCfg(**payload.model_cfg)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Diffusion checkpoint {path}: {error}") from error
    model = DiffusionTransformer(
        obs_dim=payload.obs_dim,
        action_dim=payload.action_dim,
        cfg=model_cfg,
    )
    model.load_state_dict(payload.state_dict)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        config={"dataset": payload.dataset},
        stats=payload.stats,
        meta=payload.meta,
        checkpoint_format=payload.checkpoint_format,
        robot_asset=payload.robot_asset,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "LoadedCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
]
