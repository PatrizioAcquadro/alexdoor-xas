"""Validated ACT configuration without simulator or torch imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.dataset.loader import OBS_PRESETS
from alexdoor_xas.policies.common.config import config_from_mapping, load_config
from alexdoor_xas.policies.common.types import PolicyDatasetCfg, PolicyRunCfg

VALID_SPACES = {A2_EE_DELTA, A3_OBJ_REL_EE_DELTA}


class ActConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ActModelCfg:
    chunk_size: int = 40
    d_model: int = 128
    n_heads: int = 4
    dim_feedforward: int = 512
    z_dim: int = 16
    cvae_encoder_layers: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class ActTrainCfg:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    kl_weight: float = 10.0
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    val_every: int = 5
    overfit_episodes: int | None = None


@dataclass(frozen=True)
class ActRolloutCfg:
    max_ticks: int = 600
    success_angle_deg: float = 45.0
    temporal_ensemble: bool = False
    ensemble_m: float = 0.01
    policy_device: str = "cuda"


@dataclass(frozen=True)
class ActConfig:
    dataset: PolicyDatasetCfg
    model: ActModelCfg
    train: ActTrainCfg
    run: PolicyRunCfg
    rollout: ActRolloutCfg


def load_act_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ActConfig:
    config = load_config(
        paths.REPO_ROOT / "configs" / "act.yaml",
        ActConfig,
        ActConfigError,
        hydra_overrides,
        cli_overrides,
    )
    return _validate(config)


def act_config_from_dict(payload: dict[str, Any]) -> ActConfig:
    return _validate(config_from_mapping(payload, ActConfig, ActConfigError))


def _validate(config: ActConfig) -> ActConfig:
    dataset, model, train, rollout = config.dataset, config.model, config.train, config.rollout
    if dataset.space not in VALID_SPACES:
        raise ActConfigError(f"dataset.space must be one of {sorted(VALID_SPACES)}")
    if dataset.obs_preset not in OBS_PRESETS:
        raise ActConfigError(f"dataset.obs_preset must be one of {sorted(OBS_PRESETS)}")
    if not dataset.task or not dataset.version:
        raise ActConfigError("dataset.task and dataset.version must be non-empty")
    for name in (
        "chunk_size",
        "d_model",
        "n_heads",
        "dim_feedforward",
        "z_dim",
        "cvae_encoder_layers",
        "encoder_layers",
        "decoder_layers",
    ):
        if getattr(model, name) <= 0:
            raise ActConfigError(f"model.{name} must be positive")
    if model.d_model % model.n_heads:
        raise ActConfigError("model.d_model must be divisible by model.n_heads")
    if not 0.0 <= model.dropout < 1.0:
        raise ActConfigError("model.dropout must be in [0, 1)")
    for name in ("epochs", "batch_size", "val_every"):
        if getattr(train, name) <= 0:
            raise ActConfigError(f"train.{name} must be positive")
    if train.overfit_episodes is not None and train.overfit_episodes <= 0:
        raise ActConfigError("train.overfit_episodes must be positive or null")
    for name, allow_zero in (
        ("lr", False),
        ("weight_decay", True),
        ("kl_weight", True),
        ("grad_clip", False),
    ):
        value = getattr(train, name)
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            bound = "non-negative" if allow_zero else "positive"
            raise ActConfigError(f"train.{name} must be finite and {bound}")
    if not train.device or not rollout.policy_device:
        raise ActConfigError("policy devices must be non-empty")
    if rollout.max_ticks <= 0:
        raise ActConfigError("rollout.max_ticks must be positive")
    if not math.isfinite(rollout.success_angle_deg) or rollout.success_angle_deg <= 0.0:
        raise ActConfigError("rollout.success_angle_deg must be positive and finite")
    if not math.isfinite(rollout.ensemble_m) or rollout.ensemble_m <= 0.0:
        raise ActConfigError("rollout.ensemble_m must be positive and finite")
    return config
