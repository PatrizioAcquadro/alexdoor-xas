"""Validated Diffusion Policy configuration without runtime imports."""

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
VALID_SAMPLERS = {"ddpm", "ddim"}


class DiffusionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DiffusionModelCfg:
    horizon: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"


@dataclass(frozen=True)
class DiffusionTrainCfg:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-3
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.999
    seed: int = 0
    device: str = "cuda"
    val_every: int = 10
    val_inference_steps: int = 10
    overfit_episodes: int | None = None


@dataclass(frozen=True)
class DiffusionRolloutCfg:
    max_ticks: int = 600
    success_angle_deg: float = 45.0
    n_action_steps: int = 8
    sampler: str = "ddpm"
    num_inference_steps: int = 100
    policy_device: str = "cuda"


@dataclass(frozen=True)
class DiffusionConfig:
    dataset: PolicyDatasetCfg
    model: DiffusionModelCfg
    train: DiffusionTrainCfg
    run: PolicyRunCfg
    rollout: DiffusionRolloutCfg


def load_diffusion_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> DiffusionConfig:
    config = load_config(
        paths.REPO_ROOT / "configs" / "diffusion.yaml",
        DiffusionConfig,
        DiffusionConfigError,
        hydra_overrides,
        cli_overrides,
    )
    return _validate(config)


def diffusion_config_from_dict(payload: dict[str, Any]) -> DiffusionConfig:
    return _validate(config_from_mapping(payload, DiffusionConfig, DiffusionConfigError))


def _validate(config: DiffusionConfig) -> DiffusionConfig:
    dataset, model, train, rollout = config.dataset, config.model, config.train, config.rollout
    if dataset.space not in VALID_SPACES:
        raise DiffusionConfigError(f"dataset.space must be one of {sorted(VALID_SPACES)}")
    if dataset.obs_preset not in OBS_PRESETS:
        raise DiffusionConfigError(f"dataset.obs_preset must be one of {sorted(OBS_PRESETS)}")
    if not dataset.task or not dataset.version:
        raise DiffusionConfigError("dataset.task and dataset.version must be non-empty")
    for name in (
        "horizon",
        "d_model",
        "n_heads",
        "n_decoder_layers",
        "dim_feedforward",
        "num_train_timesteps",
    ):
        if getattr(model, name) <= 0:
            raise DiffusionConfigError(f"model.{name} must be positive")
    if model.d_model % model.n_heads:
        raise DiffusionConfigError("model.d_model must be divisible by model.n_heads")
    if not 0.0 <= model.dropout < 1.0:
        raise DiffusionConfigError("model.dropout must be in [0, 1)")
    if model.beta_schedule != "squaredcos_cap_v2":
        raise DiffusionConfigError("model.beta_schedule must be squaredcos_cap_v2")
    if model.prediction_type != "epsilon":
        raise DiffusionConfigError("model.prediction_type must be epsilon")
    for name in ("epochs", "batch_size", "val_every", "val_inference_steps"):
        if getattr(train, name) <= 0:
            raise DiffusionConfigError(f"train.{name} must be positive")
    if train.lr_warmup_steps < 0:
        raise DiffusionConfigError("train.lr_warmup_steps must be non-negative")
    if train.overfit_episodes is not None and train.overfit_episodes <= 0:
        raise DiffusionConfigError("train.overfit_episodes must be positive or null")
    for name, allow_zero in (("lr", False), ("weight_decay", True), ("grad_clip", False)):
        value = getattr(train, name)
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            bound = "non-negative" if allow_zero else "positive"
            raise DiffusionConfigError(f"train.{name} must be finite and {bound}")
    if train.lr_schedule not in {"constant", "cosine"}:
        raise DiffusionConfigError("train.lr_schedule must be constant or cosine")
    if not 0.0 < train.ema_decay < 1.0:
        raise DiffusionConfigError("train.ema_decay must be in (0, 1)")
    if not train.device or not rollout.policy_device:
        raise DiffusionConfigError("policy devices must be non-empty")
    for name in ("max_ticks", "n_action_steps", "num_inference_steps"):
        if getattr(rollout, name) <= 0:
            raise DiffusionConfigError(f"rollout.{name} must be positive")
    if not math.isfinite(rollout.success_angle_deg) or rollout.success_angle_deg <= 0.0:
        raise DiffusionConfigError("rollout.success_angle_deg must be positive and finite")
    if rollout.sampler not in VALID_SAMPLERS:
        raise DiffusionConfigError("rollout.sampler must be ddpm or ddim")
    if rollout.n_action_steps > model.horizon:
        raise DiffusionConfigError("rollout.n_action_steps must not exceed model.horizon")
    if rollout.num_inference_steps > model.num_train_timesteps:
        raise DiffusionConfigError(
            "rollout.num_inference_steps must not exceed model.num_train_timesteps"
        )
    if train.val_inference_steps > model.num_train_timesteps:
        raise DiffusionConfigError(
            "train.val_inference_steps must not exceed model.num_train_timesteps"
        )
    return config
