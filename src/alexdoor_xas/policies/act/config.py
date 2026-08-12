"""Hydra/OmegaConf config loading for the ACT baseline (Phase 3.2).

This module is intentionally Isaac-free and torch-free. It may run before
AppLauncher exists, so keep imports limited to pure Python helpers and the
frozen dataset/action contracts.
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
from alexdoor_xas.dataset import OBS_PRESETS

CONFIG_DIR: Path = paths.REPO_ROOT / "configs"
CONFIG_NAME = "act"
# A1 has no adapter-v1 execution path and A4 is symbolic per-phase intent data,
# so the trainable spaces are the two per-tick EE-delta streams.
VALID_SPACES = frozenset({A2_EE_DELTA, A3_OBJ_REL_EE_DELTA})
CONFIG_SECTIONS = frozenset({"dataset", "model", "train", "run", "rollout", "wandb"})


class ActConfigError(ValueError):
    """Raised when the ACT config cannot be resolved safely."""


@dataclass(frozen=True)
class ActDatasetCfg:
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
class ActModelCfg:
    """State-only ACT CVAE dimensions."""

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
    """Supervised training hyperparameters."""

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
class ActRunCfg:
    """Output naming for one training run."""

    output_root: str | None = None


@dataclass(frozen=True)
class ActRolloutCfg:
    """Closed-loop evaluation settings (Isaac side)."""

    max_ticks: int = 600
    success_angle_deg: float = 45.0
    temporal_ensemble: bool = False
    ensemble_m: float = 0.01
    policy_device: str = "cuda"


@dataclass(frozen=True)
class ActConfig:
    """Resolved and validated ACT config."""

    dataset: ActDatasetCfg
    model: ActModelCfg
    train: ActTrainCfg
    run: ActRunCfg
    rollout: ActRolloutCfg
    wandb_overrides: dict[str, Any]

def load_act_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ActConfig:
    """Compose and validate the ACT config.

    ``hydra_overrides`` are the unconsumed ``key=value`` command-line tokens.
    ``cli_overrides`` map dotted ``section.field`` names to values; non-``None``
    values win over Hydra, preserving argparse precedence.
    """

    overrides = list(hydra_overrides or ())
    invalid_tokens = [token for token in overrides if "=" not in token]
    if invalid_tokens:
        raise ActConfigError(
            "Hydra overrides must use key=value syntax: " + ", ".join(invalid_tokens)
        )

    config = _compose_config(overrides)
    unknown_sections = sorted(set(config) - CONFIG_SECTIONS)
    if unknown_sections:
        raise ActConfigError("unknown config section(s): " + ", ".join(unknown_sections))

    nodes = {name: dict(config.get(name) or {}) for name in CONFIG_SECTIONS}
    for dotted, value in (cli_overrides or {}).items():
        section, _, field_name = dotted.partition(".")
        if section not in CONFIG_SECTIONS or not field_name:
            raise ActConfigError(f"unknown CLI override: {dotted}")
        if value is not None:
            nodes[section][field_name] = value

    return ActConfig(
        dataset=_build_dataset_cfg(nodes["dataset"]),
        model=_build_model_cfg(nodes["model"]),
        train=_build_train_cfg(nodes["train"]),
        run=_build_run_cfg(nodes["run"]),
        rollout=_build_rollout_cfg(nodes["rollout"]),
        wandb_overrides=_build_wandb_overrides(nodes["wandb"]),
    )


def act_config_from_dict(payload: dict[str, Any]) -> ActConfig:
    """Rebuild a checked config from an immutable resolved-run payload."""
    payload = dict(payload)
    if "wandb_overrides" in payload:
        payload["wandb"] = payload.pop("wandb_overrides")
    unknown_sections = sorted(set(payload) - CONFIG_SECTIONS)
    if unknown_sections:
        raise ActConfigError("unknown config section(s): " + ", ".join(unknown_sections))
    nodes = {name: dict(payload.get(name) or {}) for name in CONFIG_SECTIONS}
    return ActConfig(
        dataset=_build_dataset_cfg(nodes["dataset"]),
        model=_build_model_cfg(nodes["model"]),
        train=_build_train_cfg(nodes["train"]),
        run=_build_run_cfg(nodes["run"]),
        rollout=_build_rollout_cfg(nodes["rollout"]),
        wandb_overrides=_build_wandb_overrides(nodes["wandb"]),
    )


def _compose_config(overrides: list[str]) -> dict[str, Any]:
    if not CONFIG_DIR.is_dir():
        raise ActConfigError(f"config directory not found: {CONFIG_DIR}")

    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as error:  # noqa: BLE001 - normalize Hydra/OmegaConf errors for the CLI.
        raise ActConfigError(str(error)) from error

    if not isinstance(container, dict):
        raise ActConfigError("Hydra config did not resolve to a mapping")
    return container


def _build_dataset_cfg(node: dict[str, Any]) -> ActDatasetCfg:
    _reject_unknown("dataset", node, {"task", "space", "version", "view_id", "obs_preset"})

    space = str(node.get("space", A2_EE_DELTA))
    if space not in VALID_SPACES:
        raise ActConfigError(
            f"dataset.space must be one of {sorted(VALID_SPACES)}, got {space!r}"
        )

    obs_preset = str(node.get("obs_preset", "core"))
    if obs_preset not in OBS_PRESETS:
        raise ActConfigError(
            f"dataset.obs_preset must be one of {sorted(OBS_PRESETS)}, got {obs_preset!r}"
        )

    task = _required_str("dataset.task", node.get("task", "door_push_alex_v2"))
    version = _required_str("dataset.version", node.get("version", "v2_pose"))
    view_id = _optional_str("dataset.view_id", node.get("view_id"))
    return ActDatasetCfg(
        task=task,
        space=space,
        version=version,
        view_id=view_id,
        obs_preset=obs_preset,
    )


def _build_model_cfg(node: dict[str, Any]) -> ActModelCfg:
    field_names = {
        "chunk_size",
        "d_model",
        "n_heads",
        "dim_feedforward",
        "z_dim",
        "cvae_encoder_layers",
        "encoder_layers",
        "decoder_layers",
        "dropout",
    }
    _reject_unknown("model", node, field_names)

    defaults = ActModelCfg()
    values: dict[str, Any] = {}
    for name in field_names - {"dropout"}:
        value = _coerce_int(f"model.{name}", node.get(name, getattr(defaults, name)))
        if value <= 0:
            raise ActConfigError(f"model.{name} must be positive")
        values[name] = value

    dropout = _coerce_float("model.dropout", node.get("dropout", defaults.dropout))
    if not (0.0 <= dropout < 1.0):
        raise ActConfigError("model.dropout must be in [0, 1)")
    values["dropout"] = dropout

    if values["d_model"] % values["n_heads"] != 0:
        raise ActConfigError("model.d_model must be divisible by model.n_heads")
    return ActModelCfg(**values)


def _build_train_cfg(node: dict[str, Any]) -> ActTrainCfg:
    field_names = {
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "kl_weight",
        "grad_clip",
        "seed",
        "device",
        "val_every",
        "overfit_episodes",
    }
    _reject_unknown("train", node, field_names)
    defaults = ActTrainCfg()

    epochs = _coerce_int("train.epochs", node.get("epochs", defaults.epochs))
    batch_size = _coerce_int("train.batch_size", node.get("batch_size", defaults.batch_size))
    val_every = _coerce_int("train.val_every", node.get("val_every", defaults.val_every))
    if epochs <= 0 or batch_size <= 0 or val_every <= 0:
        raise ActConfigError("train.epochs, train.batch_size, train.val_every must be positive")

    lr = _coerce_float("train.lr", node.get("lr", defaults.lr))
    weight_decay = _coerce_float(
        "train.weight_decay", node.get("weight_decay", defaults.weight_decay)
    )
    kl_weight = _coerce_float("train.kl_weight", node.get("kl_weight", defaults.kl_weight))
    grad_clip = _coerce_float("train.grad_clip", node.get("grad_clip", defaults.grad_clip))
    if not (math.isfinite(lr) and lr > 0.0):
        raise ActConfigError("train.lr must be a positive finite number")
    if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
        raise ActConfigError("train.weight_decay must be a non-negative finite number")
    if not (math.isfinite(kl_weight) and kl_weight >= 0.0):
        raise ActConfigError("train.kl_weight must be a non-negative finite number")
    if not (math.isfinite(grad_clip) and grad_clip > 0.0):
        raise ActConfigError("train.grad_clip must be a positive finite number")

    overfit_episodes = node.get("overfit_episodes")
    if overfit_episodes is not None:
        overfit_episodes = _coerce_int("train.overfit_episodes", overfit_episodes)
        if overfit_episodes <= 0:
            raise ActConfigError("train.overfit_episodes must be positive or null")

    return ActTrainCfg(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        kl_weight=kl_weight,
        grad_clip=grad_clip,
        seed=_coerce_int("train.seed", node.get("seed", defaults.seed)),
        device=_required_str("train.device", node.get("device", defaults.device)),
        val_every=val_every,
        overfit_episodes=overfit_episodes,
    )


def _build_run_cfg(node: dict[str, Any]) -> ActRunCfg:
    _reject_unknown("run", node, {"output_root"})
    return ActRunCfg(output_root=_optional_str("run.output_root", node.get("output_root")))


def _build_rollout_cfg(node: dict[str, Any]) -> ActRolloutCfg:
    field_names = {
        "max_ticks",
        "success_angle_deg",
        "temporal_ensemble",
        "ensemble_m",
        "policy_device",
    }
    _reject_unknown("rollout", node, field_names)
    defaults = ActRolloutCfg()

    max_ticks = _coerce_int("rollout.max_ticks", node.get("max_ticks", defaults.max_ticks))
    if max_ticks <= 0:
        raise ActConfigError("rollout.max_ticks must be positive")

    success_angle_deg = _coerce_float(
        "rollout.success_angle_deg",
        node.get("success_angle_deg", defaults.success_angle_deg),
    )
    if not (math.isfinite(success_angle_deg) and success_angle_deg > 0.0):
        raise ActConfigError("rollout.success_angle_deg must be a positive finite number")

    ensemble_m = _coerce_float("rollout.ensemble_m", node.get("ensemble_m", defaults.ensemble_m))
    if not (math.isfinite(ensemble_m) and ensemble_m > 0.0):
        raise ActConfigError("rollout.ensemble_m must be a positive finite number")

    return ActRolloutCfg(
        max_ticks=max_ticks,
        success_angle_deg=success_angle_deg,
        temporal_ensemble=_coerce_bool(
            "rollout.temporal_ensemble",
            node.get("temporal_ensemble", defaults.temporal_ensemble),
        ),
        ensemble_m=ensemble_m,
        policy_device=_required_str(
            "rollout.policy_device", node.get("policy_device", defaults.policy_device)
        ),
    )


def _build_wandb_overrides(node: dict[str, Any]) -> dict[str, Any]:
    # Field names are validated downstream by WandbConfig.from_mapping.
    return {key: value for key, value in node.items() if value is not None}


def _reject_unknown(section: str, node: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(node) - known)
    if unknown:
        raise ActConfigError(f"unknown {section} config field(s): " + ", ".join(unknown))


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ActConfigError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ActConfigError(f"{name} must be an integer") from error
    if integer != value and not (isinstance(value, float) and value.is_integer()):
        raise ActConfigError(f"{name} must be an integer")
    return integer


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ActConfigError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ActConfigError(f"{name} must be a number") from error


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ActConfigError(f"{name} must be a boolean")


def _required_str(name: str, value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    raise ActConfigError(f"{name} must be a non-empty string")


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ActConfigError(f"{name} must be a string or null")


__all__ = [
    "CONFIG_DIR",
    "CONFIG_NAME",
    "VALID_SPACES",
    "ActConfig",
    "ActConfigError",
    "ActDatasetCfg",
    "ActModelCfg",
    "ActRolloutCfg",
    "ActRunCfg",
    "ActTrainCfg",
    "act_config_from_dict",
    "load_act_config",
]
