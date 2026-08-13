"""Hydra/OmegaConf config loading for the Diffusion Policy baseline (Phase 3.3).

This module is intentionally Isaac-free, torch-free, and diffusers-free. It
may run before AppLauncher exists, so keep imports limited to pure Python
helpers and the frozen dataset/action contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.dataset.loader import OBS_PRESETS

CONFIG_DIR: Path = paths.REPO_ROOT / "configs"
CONFIG_NAME = "diffusion"
# A1 has no adapter-v1 execution path and A4 is symbolic per-phase intent data,
# so the trainable spaces are the two per-tick EE-delta streams (same as ACT).
VALID_SPACES = frozenset({A2_EE_DELTA, A3_OBJ_REL_EE_DELTA})
CONFIG_SECTIONS = frozenset({"dataset", "model", "train", "run", "rollout"})

# Only the settings the checkpointed schedulers are actually rebuilt with are
# valid; widening these means widening the checkpoint contract too.
VALID_BETA_SCHEDULES = frozenset({"squaredcos_cap_v2"})
VALID_PREDICTION_TYPES = frozenset({"epsilon"})
VALID_SAMPLERS = frozenset({"ddpm", "ddim"})
VALID_LR_SCHEDULES = frozenset({"constant", "cosine"})


class DiffusionConfigError(ValueError):
    """Raised when the diffusion config cannot be resolved safely."""


@dataclass(frozen=True)
class DiffusionDatasetCfg:
    """Which frozen dataset export the trainer consumes."""

    task: str = "door_push_alex_v2"
    space: str = A2_EE_DELTA
    version: str = "v2_pose"
    view_id: str | None = None
    obs_preset: str = "core"

    @property
    def dataset_dir(self) -> Path:
        return paths.DATASETS_DIR / self.task / self.space / self.version


@dataclass(frozen=True)
class DiffusionModelCfg:
    """Time-series diffusion transformer dimensions plus the noise schedule.

    The schedule lives with the model because a checkpoint must rebuild the
    exact training schedule to sample correctly.
    """

    horizon: int = 16  # Tp: predicted chunk length
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
    """Denoising training hyperparameters."""

    epochs: int = 300
    batch_size: int = 64
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-3
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"  # transformer DP is tuning-sensitive; paper uses warmup+cosine
    lr_warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.999
    seed: int = 0
    device: str = "cuda"  # GPU-first; verification gates may override to cpu
    val_every: int = 10
    val_inference_steps: int = 10  # DDIM steps for the sampled-L1 val metric
    overfit_episodes: int | None = None


@dataclass(frozen=True)
class DiffusionRunCfg:
    """Output naming for one training run."""

    output_root: str | None = None


@dataclass(frozen=True)
class DiffusionRolloutCfg:
    """Closed-loop evaluation settings (Isaac side)."""

    max_ticks: int = 600
    success_angle_deg: float = 45.0
    n_action_steps: int = 8  # Ta: executed prefix of each predicted chunk (receding horizon)
    sampler: str = "ddpm"
    num_inference_steps: int = 100
    policy_device: str = "cuda"


@dataclass(frozen=True)
class DiffusionConfig:
    """Resolved and validated diffusion config."""

    dataset: DiffusionDatasetCfg
    model: DiffusionModelCfg
    train: DiffusionTrainCfg
    run: DiffusionRunCfg
    rollout: DiffusionRolloutCfg


def load_diffusion_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> DiffusionConfig:
    """Compose and validate the diffusion config.

    ``hydra_overrides`` are the unconsumed ``key=value`` command-line tokens.
    ``cli_overrides`` map dotted ``section.field`` names to values; non-``None``
    values win over Hydra, preserving argparse precedence.
    """

    overrides = list(hydra_overrides or ())
    invalid_tokens = [token for token in overrides if "=" not in token]
    if invalid_tokens:
        raise DiffusionConfigError(
            "Hydra overrides must use key=value syntax: " + ", ".join(invalid_tokens)
        )

    config = _compose_config(overrides)
    unknown_sections = sorted(set(config) - CONFIG_SECTIONS)
    if unknown_sections:
        raise DiffusionConfigError("unknown config section(s): " + ", ".join(unknown_sections))

    nodes = {name: dict(config.get(name) or {}) for name in CONFIG_SECTIONS}
    for dotted, value in (cli_overrides or {}).items():
        section, _, field_name = dotted.partition(".")
        if section not in CONFIG_SECTIONS or not field_name:
            raise DiffusionConfigError(f"unknown CLI override: {dotted}")
        if value is not None:
            nodes[section][field_name] = value

    resolved = DiffusionConfig(
        dataset=_build_dataset_cfg(nodes["dataset"]),
        model=_build_model_cfg(nodes["model"]),
        train=_build_train_cfg(nodes["train"]),
        run=_build_run_cfg(nodes["run"]),
        rollout=_build_rollout_cfg(nodes["rollout"]),
    )

    return _validate_cross_section(resolved)


def diffusion_config_from_dict(payload: dict[str, Any]) -> DiffusionConfig:
    """Rebuild a checked config from an immutable resolved-run payload."""
    payload = dict(payload)
    unknown_sections = sorted(set(payload) - CONFIG_SECTIONS)
    if unknown_sections:
        raise DiffusionConfigError("unknown config section(s): " + ", ".join(unknown_sections))
    nodes = {name: dict(payload.get(name) or {}) for name in CONFIG_SECTIONS}
    return _validate_cross_section(
        DiffusionConfig(
            dataset=_build_dataset_cfg(nodes["dataset"]),
            model=_build_model_cfg(nodes["model"]),
            train=_build_train_cfg(nodes["train"]),
            run=_build_run_cfg(nodes["run"]),
            rollout=_build_rollout_cfg(nodes["rollout"]),
        )
    )


def _validate_cross_section(resolved: DiffusionConfig) -> DiffusionConfig:
    # Cross-section constraints need both nodes resolved.
    if resolved.rollout.n_action_steps > resolved.model.horizon:
        raise DiffusionConfigError(
            f"rollout.n_action_steps ({resolved.rollout.n_action_steps}) must not exceed "
            f"model.horizon ({resolved.model.horizon})"
        )
    if resolved.rollout.num_inference_steps > resolved.model.num_train_timesteps:
        raise DiffusionConfigError(
            f"rollout.num_inference_steps ({resolved.rollout.num_inference_steps}) must not "
            f"exceed model.num_train_timesteps ({resolved.model.num_train_timesteps})"
        )
    if resolved.train.val_inference_steps > resolved.model.num_train_timesteps:
        raise DiffusionConfigError(
            f"train.val_inference_steps ({resolved.train.val_inference_steps}) must not "
            f"exceed model.num_train_timesteps ({resolved.model.num_train_timesteps})"
        )
    return resolved


def _compose_config(overrides: list[str]) -> dict[str, Any]:
    if not CONFIG_DIR.is_dir():
        raise DiffusionConfigError(f"config directory not found: {CONFIG_DIR}")

    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as error:  # noqa: BLE001 - normalize Hydra/OmegaConf errors for the CLI.
        raise DiffusionConfigError(str(error)) from error

    if not isinstance(container, dict):
        raise DiffusionConfigError("Hydra config did not resolve to a mapping")
    return container


def _build_dataset_cfg(node: dict[str, Any]) -> DiffusionDatasetCfg:
    _reject_unknown("dataset", node, {"task", "space", "version", "view_id", "obs_preset"})

    space = str(node.get("space", A2_EE_DELTA))
    if space not in VALID_SPACES:
        raise DiffusionConfigError(
            f"dataset.space must be one of {sorted(VALID_SPACES)}, got {space!r}"
        )

    obs_preset = str(node.get("obs_preset", "core"))
    if obs_preset not in OBS_PRESETS:
        raise DiffusionConfigError(
            f"dataset.obs_preset must be one of {sorted(OBS_PRESETS)}, got {obs_preset!r}"
        )

    task = _required_str("dataset.task", node.get("task", "door_push_alex_v2"))
    version = _required_str("dataset.version", node.get("version", "v2_pose"))
    view_id = _optional_str("dataset.view_id", node.get("view_id"))
    return DiffusionDatasetCfg(
        task=task,
        space=space,
        version=version,
        view_id=view_id,
        obs_preset=obs_preset,
    )


def _build_model_cfg(node: dict[str, Any]) -> DiffusionModelCfg:
    field_names = {
        "horizon",
        "d_model",
        "n_heads",
        "n_decoder_layers",
        "dim_feedforward",
        "dropout",
        "num_train_timesteps",
        "beta_schedule",
        "prediction_type",
    }
    _reject_unknown("model", node, field_names)

    defaults = DiffusionModelCfg()
    int_fields = {
        "horizon",
        "d_model",
        "n_heads",
        "n_decoder_layers",
        "dim_feedforward",
        "num_train_timesteps",
    }
    values: dict[str, Any] = {}
    for name in int_fields:
        value = _coerce_int(f"model.{name}", node.get(name, getattr(defaults, name)))
        if value <= 0:
            raise DiffusionConfigError(f"model.{name} must be positive")
        values[name] = value

    dropout = _coerce_float("model.dropout", node.get("dropout", defaults.dropout))
    if not (0.0 <= dropout < 1.0):
        raise DiffusionConfigError("model.dropout must be in [0, 1)")
    values["dropout"] = dropout

    beta_schedule = _required_str(
        "model.beta_schedule", node.get("beta_schedule", defaults.beta_schedule)
    )
    if beta_schedule not in VALID_BETA_SCHEDULES:
        raise DiffusionConfigError(
            f"model.beta_schedule must be one of {sorted(VALID_BETA_SCHEDULES)}, "
            f"got {beta_schedule!r}"
        )
    values["beta_schedule"] = beta_schedule

    prediction_type = _required_str(
        "model.prediction_type", node.get("prediction_type", defaults.prediction_type)
    )
    if prediction_type not in VALID_PREDICTION_TYPES:
        raise DiffusionConfigError(
            f"model.prediction_type must be one of {sorted(VALID_PREDICTION_TYPES)}, "
            f"got {prediction_type!r}"
        )
    values["prediction_type"] = prediction_type

    if values["d_model"] % values["n_heads"] != 0:
        raise DiffusionConfigError("model.d_model must be divisible by model.n_heads")
    return DiffusionModelCfg(**values)


def _build_train_cfg(node: dict[str, Any]) -> DiffusionTrainCfg:
    field_names = {
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "lr_schedule",
        "lr_warmup_steps",
        "use_ema",
        "ema_decay",
        "seed",
        "device",
        "val_every",
        "val_inference_steps",
        "overfit_episodes",
    }
    _reject_unknown("train", node, field_names)
    defaults = DiffusionTrainCfg()

    epochs = _coerce_int("train.epochs", node.get("epochs", defaults.epochs))
    batch_size = _coerce_int("train.batch_size", node.get("batch_size", defaults.batch_size))
    val_every = _coerce_int("train.val_every", node.get("val_every", defaults.val_every))
    val_inference_steps = _coerce_int(
        "train.val_inference_steps",
        node.get("val_inference_steps", defaults.val_inference_steps),
    )
    if epochs <= 0 or batch_size <= 0 or val_every <= 0 or val_inference_steps <= 0:
        raise DiffusionConfigError(
            "train.epochs, train.batch_size, train.val_every, "
            "train.val_inference_steps must be positive"
        )

    lr = _coerce_float("train.lr", node.get("lr", defaults.lr))
    weight_decay = _coerce_float(
        "train.weight_decay", node.get("weight_decay", defaults.weight_decay)
    )
    grad_clip = _coerce_float("train.grad_clip", node.get("grad_clip", defaults.grad_clip))
    ema_decay = _coerce_float("train.ema_decay", node.get("ema_decay", defaults.ema_decay))
    if not (math.isfinite(lr) and lr > 0.0):
        raise DiffusionConfigError("train.lr must be a positive finite number")
    if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
        raise DiffusionConfigError("train.weight_decay must be a non-negative finite number")
    if not (math.isfinite(grad_clip) and grad_clip > 0.0):
        raise DiffusionConfigError("train.grad_clip must be a positive finite number")
    if not (0.0 < ema_decay < 1.0):
        raise DiffusionConfigError("train.ema_decay must be in (0, 1)")

    lr_schedule = _required_str("train.lr_schedule", node.get("lr_schedule", defaults.lr_schedule))
    if lr_schedule not in VALID_LR_SCHEDULES:
        raise DiffusionConfigError(
            f"train.lr_schedule must be one of {sorted(VALID_LR_SCHEDULES)}, got {lr_schedule!r}"
        )

    lr_warmup_steps = _coerce_int(
        "train.lr_warmup_steps", node.get("lr_warmup_steps", defaults.lr_warmup_steps)
    )
    if lr_warmup_steps < 0:
        raise DiffusionConfigError("train.lr_warmup_steps must be non-negative")

    overfit_episodes = node.get("overfit_episodes")
    if overfit_episodes is not None:
        overfit_episodes = _coerce_int("train.overfit_episodes", overfit_episodes)
        if overfit_episodes <= 0:
            raise DiffusionConfigError("train.overfit_episodes must be positive or null")

    return DiffusionTrainCfg(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        lr_schedule=lr_schedule,
        lr_warmup_steps=lr_warmup_steps,
        use_ema=_coerce_bool("train.use_ema", node.get("use_ema", defaults.use_ema)),
        ema_decay=ema_decay,
        seed=_coerce_int("train.seed", node.get("seed", defaults.seed)),
        device=_required_str("train.device", node.get("device", defaults.device)),
        val_every=val_every,
        val_inference_steps=val_inference_steps,
        overfit_episodes=overfit_episodes,
    )


def _build_run_cfg(node: dict[str, Any]) -> DiffusionRunCfg:
    _reject_unknown("run", node, {"output_root"})
    return DiffusionRunCfg(
        output_root=_optional_str("run.output_root", node.get("output_root")),
    )


def _build_rollout_cfg(node: dict[str, Any]) -> DiffusionRolloutCfg:
    field_names = {
        "max_ticks",
        "success_angle_deg",
        "n_action_steps",
        "sampler",
        "num_inference_steps",
        "policy_device",
    }
    _reject_unknown("rollout", node, field_names)
    defaults = DiffusionRolloutCfg()

    max_ticks = _coerce_int("rollout.max_ticks", node.get("max_ticks", defaults.max_ticks))
    if max_ticks <= 0:
        raise DiffusionConfigError("rollout.max_ticks must be positive")

    success_angle_deg = _coerce_float(
        "rollout.success_angle_deg",
        node.get("success_angle_deg", defaults.success_angle_deg),
    )
    if not (math.isfinite(success_angle_deg) and success_angle_deg > 0.0):
        raise DiffusionConfigError("rollout.success_angle_deg must be a positive finite number")

    n_action_steps = _coerce_int(
        "rollout.n_action_steps", node.get("n_action_steps", defaults.n_action_steps)
    )
    if n_action_steps <= 0:
        raise DiffusionConfigError("rollout.n_action_steps must be positive")

    sampler = _required_str("rollout.sampler", node.get("sampler", defaults.sampler))
    if sampler not in VALID_SAMPLERS:
        raise DiffusionConfigError(
            f"rollout.sampler must be one of {sorted(VALID_SAMPLERS)}, got {sampler!r}"
        )

    num_inference_steps = _coerce_int(
        "rollout.num_inference_steps",
        node.get("num_inference_steps", defaults.num_inference_steps),
    )
    if num_inference_steps <= 0:
        raise DiffusionConfigError("rollout.num_inference_steps must be positive")

    return DiffusionRolloutCfg(
        max_ticks=max_ticks,
        success_angle_deg=success_angle_deg,
        n_action_steps=n_action_steps,
        sampler=sampler,
        num_inference_steps=num_inference_steps,
        policy_device=_required_str(
            "rollout.policy_device", node.get("policy_device", defaults.policy_device)
        ),
    )


def _reject_unknown(section: str, node: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(node) - known)
    if unknown:
        raise DiffusionConfigError(f"unknown {section} config field(s): " + ", ".join(unknown))


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise DiffusionConfigError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise DiffusionConfigError(f"{name} must be an integer") from error
    if integer != value and not (isinstance(value, float) and value.is_integer()):
        raise DiffusionConfigError(f"{name} must be an integer")
    return integer


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise DiffusionConfigError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise DiffusionConfigError(f"{name} must be a number") from error


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise DiffusionConfigError(f"{name} must be a boolean")


def _required_str(name: str, value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    raise DiffusionConfigError(f"{name} must be a non-empty string")


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise DiffusionConfigError(f"{name} must be a string or null")


__all__ = [
    "CONFIG_DIR",
    "CONFIG_NAME",
    "VALID_BETA_SCHEDULES",
    "VALID_LR_SCHEDULES",
    "VALID_PREDICTION_TYPES",
    "VALID_SAMPLERS",
    "VALID_SPACES",
    "DiffusionConfig",
    "DiffusionConfigError",
    "DiffusionDatasetCfg",
    "DiffusionModelCfg",
    "DiffusionRolloutCfg",
    "DiffusionRunCfg",
    "DiffusionTrainCfg",
    "diffusion_config_from_dict",
    "load_diffusion_config",
]
