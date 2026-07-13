"""Self-contained ACT checkpoints.

A checkpoint carries everything needed to rebuild and run the policy without
the training dataset on disk: model weights and dimensions, the resolved run
config, and the normalization stats the model was trained with (JSON-style
payload so ``torch.load(weights_only=True)`` stays safe).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset import DatasetNormStats, NormStats
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel

CHECKPOINT_FORMAT = "alexdoor_xas.act.v1"


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A rebuilt ACT model plus the context it was trained in."""

    model: ACTModel
    config: dict[str, Any]
    stats: DatasetNormStats
    meta: dict[str, Any]
    split_episode_ids: dict[str, tuple[str, ...]]
    provenance: dict[str, Any]
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
    split_episode_ids: dict[str, list[str] | tuple[str, ...]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained checkpoint; returns the written path."""
    _validate_dataset_binding(config, stats)
    _validate_training_provenance(config, stats, split_episode_ids, provenance)
    if _is_v2_config(config) and robot_asset is None:
        raise ValueError("Alex V2 checkpoints require robot asset provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "model_cfg": asdict(model.cfg),
        "config": config,
        "norm_stats": _stats_payload(stats),
        "split_episode_ids": _split_payload(split_episode_ids),
        "provenance": dict(provenance or {}),
        "robot_asset": robot_asset.to_dict() if robot_asset is not None else None,
        "meta": {**(meta or {}), "torch_version": str(torch.__version__)},
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
    model = ACTModel(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        cfg=ActModelCfg(**payload["model_cfg"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    config = dict(payload["config"])
    robot_asset = _asset_from_payload(payload.get("robot_asset"))
    if _is_v2_config(config) and robot_asset is None:
        raise ValueError("Alex V2 checkpoint is missing robot asset provenance")
    stats = _stats_from_payload(payload["norm_stats"])
    split_ids = _split_from_payload(payload.get("split_episode_ids"))
    provenance = dict(payload.get("provenance") or {})
    _validate_training_provenance(config, stats, split_ids, provenance)
    return LoadedCheckpoint(
        model=model,
        config=config,
        stats=stats,
        meta=dict(payload["meta"]),
        split_episode_ids=split_ids,
        provenance=provenance,
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


def _validate_dataset_binding(config: dict[str, Any], stats: DatasetNormStats) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("checkpoint config requires an embedded dataset mapping")
    if dataset.get("space") != stats.action_space:
        raise ValueError("checkpoint dataset action space does not match norm stats")
    if dataset.get("obs_preset") != stats.obs_preset:
        raise ValueError("checkpoint dataset observation preset does not match norm stats")


def _validate_training_provenance(
    config: dict[str, Any],
    stats: DatasetNormStats,
    split_episode_ids: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> None:
    dataset = config.get("dataset") or {}
    view_id = dataset.get("view_id") if isinstance(dataset, dict) else None
    if view_id is None:
        return
    if not isinstance(provenance, dict) or provenance.get("schema") not in {
        "alexdoor_xas.training_provenance.v1",
        "alexdoor_xas.training_provenance.v2",
    }:
        raise ValueError("view checkpoint requires training provenance")
    schema = provenance["schema"]
    if str(view_id).startswith("v3_scale_n") and schema != "alexdoor_xas.training_provenance.v2":
        raise ValueError("scale view checkpoint requires dual-fingerprint provenance v2")
    expected_splits = _split_payload(split_episode_ids) or {}
    required = {
        "view_id": view_id,
        "view_fingerprint_sha256": stats.view_fingerprint,
        "split_episode_ids": expected_splits,
        "split_counts": {
            name: len(expected_splits.get(name, ()))
            for name in ("train", "val", "test")
        },
        "normalization_fingerprint_sha256": stats.normalization_fingerprint,
        "normalization_sha256": stats.normalization_sha256,
        "action_space": stats.action_space,
        "obs_preset": stats.obs_preset,
    }
    for key, value in required.items():
        if provenance.get(key) != value:
            raise ValueError(f"checkpoint training provenance mismatch: {key}")
    from alexdoor_xas.cluster_sweep.config import canonical_resolved_config_sha256

    split_digest = canonical_resolved_config_sha256(expected_splits)
    if provenance.get("split_fingerprint_sha256") != split_digest:
        raise ValueError("checkpoint training provenance mismatch: split fingerprint")
    config_digest = canonical_resolved_config_sha256(config)
    if provenance.get("resolved_training_config_sha256") != config_digest:
        raise ValueError("checkpoint training provenance mismatch: resolved config")
    for key, pattern in (
        ("normalization_sha256", r"[0-9a-f]{64}"),
        ("source_git_commit", r"[0-9a-f]{40}"),
    ):
        if re.fullmatch(pattern, str(provenance.get(key, ""))) is None:
            raise ValueError(f"checkpoint training provenance has invalid {key}")
    if provenance.get("normalization_sha256") != stats.normalization_sha256:
        raise ValueError("checkpoint training provenance mismatch: normalization_sha256")
    if schema == "alexdoor_xas.training_provenance.v2":
        master = str(provenance.get("master_dataset_fingerprint_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", master) is None:
            raise ValueError(
                "checkpoint training provenance mismatch: "
                "master_dataset_fingerprint_sha256"
            )
        if provenance.get("action_dataset_fingerprint_sha256") != stats.dataset_fingerprint:
            raise ValueError(
                "checkpoint training provenance mismatch: "
                "action_dataset_fingerprint_sha256"
            )
    elif provenance.get("master_dataset_fingerprint_sha256") != stats.dataset_fingerprint:
        raise ValueError(
            "checkpoint training provenance mismatch: master_dataset_fingerprint_sha256"
        )


def _stats_payload(stats: DatasetNormStats) -> dict[str, Any]:
    # Mirrors dataset.normalize.save_norm_stats so the embedded copy stays
    # byte-compatible with the on-disk norm_stats.json layout.
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
        "normalization_fingerprint_sha256": stats.normalization_fingerprint,
        "normalization_sha256": stats.normalization_sha256,
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
        view_id=str(payload["view_id"]) if payload.get("view_id") is not None else None,
        view_fingerprint=str(payload.get("view_fingerprint_sha256", "")),
        normalization_fingerprint=str(
            payload.get("normalization_fingerprint_sha256", "")
        ),
        normalization_sha256=str(payload.get("normalization_sha256", "")),
    )


__all__ = ["CHECKPOINT_FORMAT", "LoadedCheckpoint", "load_checkpoint", "save_checkpoint"]
