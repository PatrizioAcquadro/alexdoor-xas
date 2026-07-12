"""Self-contained Diffusion Policy checkpoints.

A checkpoint carries everything needed to rebuild and run the policy without
the training dataset on disk: model weights (the EMA shadow when EMA was on —
the deployed policy), the model config including the noise-schedule fields
(so the exact training schedule and samplers rebuild from the checkpoint
alone), the resolved run config, and the normalization stats (JSON-style
payload so ``torch.load(weights_only=True)`` stays safe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset import DatasetNormStats, NormStats
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.diffusion.model import DiffusionTransformer
from alexdoor_xas.policies.diffusion.schedulers import scheduler_config_payload

CHECKPOINT_FORMAT = "alexdoor_xas.diffusion.v1"


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A rebuilt diffusion model plus the context it was trained in."""

    model: DiffusionTransformer
    config: dict[str, Any]
    stats: DatasetNormStats
    meta: dict[str, Any]
    split_episode_ids: dict[str, tuple[str, ...]]
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
    split_episode_ids: dict[str, list[str] | tuple[str, ...]] | None = None,
) -> Path:
    """Write a self-contained checkpoint; returns the written path."""
    if _is_v2_config(config) and robot_asset is None:
        raise ValueError("Alex V2 checkpoints require robot asset provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import diffusers

        diffusers_version = str(diffusers.__version__)
    except ImportError:  # pragma: no cover - diffusers is a hard runtime dep
        diffusers_version = "unknown"
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "model_cfg": asdict(model.cfg),
        "scheduler": scheduler_config_payload(model.cfg),
        "config": config,
        "norm_stats": _stats_payload(stats),
        "split_episode_ids": _split_payload(split_episode_ids),
        "robot_asset": robot_asset.to_dict() if robot_asset is not None else None,
        "meta": {
            **(meta or {}),
            "torch_version": str(torch.__version__),
            "diffusers_version": diffusers_version,
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """Rebuild the model (eval mode) and its training context from disk."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format')!r} in {path}"
        )
    model = DiffusionTransformer(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        cfg=DiffusionModelCfg(**payload["model_cfg"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    config = dict(payload["config"])
    robot_asset = _asset_from_payload(payload.get("robot_asset"))
    if _is_v2_config(config) and robot_asset is None:
        raise ValueError("Alex V2 checkpoint is missing robot asset provenance")
    return LoadedCheckpoint(
        model=model,
        config=config,
        stats=_stats_from_payload(payload["norm_stats"]),
        meta=dict(payload["meta"]),
        split_episode_ids=_split_from_payload(payload.get("split_episode_ids")),
        robot_asset=robot_asset,
    )


def _asset_from_payload(payload: Any) -> RobotAssetRef | None:
    return None if payload is None else RobotAssetRef.from_dict(payload)


def _split_payload(
    split_episode_ids: dict[str, list[str] | tuple[str, ...]] | None,
) -> dict[str, list[str]] | None:
    if split_episode_ids is None:
        return None
    return {
        name: [str(episode_id) for episode_id in split_episode_ids.get(name, ())]
        for name in ("train", "val", "test")
    }


def _split_from_payload(payload: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(payload, dict):
        return {}
    return {
        name: tuple(str(episode_id) for episode_id in payload.get(name, ()))
        for name in ("train", "val", "test")
    }


def _is_v2_config(config: dict[str, Any]) -> bool:
    dataset = config.get("dataset")
    return isinstance(dataset, dict) and dataset.get("task") == paths.ALEX_V2_TASK


def _stats_payload(stats: DatasetNormStats) -> dict[str, Any]:
    # Mirrors dataset.normalize.save_norm_stats so the embedded copy stays
    # byte-compatible with the on-disk norm_stats.json layout (min/max
    # included — the min-max action normalizer rebuilds from it).
    return {
        "action": stats.action.to_dict(),
        "obs": stats.obs.to_dict(),
        "obs_preset": stats.obs_preset,
        "train_episode_ids": list(stats.train_episode_ids),
        "dataset_episode_ids": list(stats.dataset_episode_ids),
        "action_space": stats.action_space,
        "dataset_fingerprint": stats.dataset_fingerprint,
        "split_name": stats.split_name,
    }


def _stats_from_payload(payload: dict[str, Any]) -> DatasetNormStats:
    return DatasetNormStats(
        action=NormStats.from_dict(payload["action"]),
        obs=NormStats.from_dict(payload["obs"]),
        obs_preset=str(payload["obs_preset"]),
        train_episode_ids=tuple(payload["train_episode_ids"]),
        dataset_episode_ids=tuple(payload["dataset_episode_ids"]),
        action_space=str(payload["action_space"]),
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        split_name=str(payload["split_name"]),
    )


__all__ = ["CHECKPOINT_FORMAT", "LoadedCheckpoint", "load_checkpoint", "save_checkpoint"]
