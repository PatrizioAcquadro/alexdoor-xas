"""Hydra/OmegaConf config loading for the scripted baseline CLI.

This module is intentionally Isaac-free. It may run before AppLauncher exists,
so keep imports limited to pure Python helpers and the scripted controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from alexdoor_xas import paths
from alexdoor_xas.policies.scripted import DoorPushControllerCfg

CONFIG_DIR: Path = paths.REPO_ROOT / "configs"
CONFIG_NAME = "scripted_baseline"
CONTROLLER_FIELD_TYPES = {
    field.name: field.type for field in fields(DoorPushControllerCfg)
}
RUN_FIELD_NAMES = frozenset(
    {
        "episodes",
        "randomized",
        "seed",
        "experiment",
        "run_id",
        "success_angle_deg",
        "max_ticks",
        "video",
        "clean_shutdown",
        "export",
        "door_yaw_deg",
        "door_offset_x",
        "door_offset_y",
        "door_pose_id",
    }
)


class ScriptedBaselineConfigError(ValueError):
    """Raised when the scripted baseline config cannot be resolved safely."""


@dataclass(frozen=True)
class ScriptedBaselineRunCfg:
    """Run-level settings for ``scripts/run_scripted_baseline.py``."""

    episodes: int = 5
    randomized: int = 0
    seed: int = 0
    experiment: str | None = None
    run_id: str | None = None
    success_angle_deg: float = 45.0
    max_ticks: int = 600
    video: bool = False
    clean_shutdown: bool = False
    export: bool = True
    door_yaw_deg: float = 0.0
    door_offset_x: float = 0.0
    door_offset_y: float = 0.0
    door_pose_id: str | None = None


@dataclass(frozen=True)
class ScriptedBaselineCfg:
    """Resolved Hydra config plus checked controller overrides."""

    run: ScriptedBaselineRunCfg
    controller_overrides: dict[str, Any]


def load_scripted_baseline_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ScriptedBaselineCfg:
    """Compose and validate the scripted baseline config.

    ``hydra_overrides`` are the unconsumed ``key=value`` command-line tokens.
    ``cli_overrides`` are argparse values; non-``None`` values win over Hydra.
    """

    overrides = list(hydra_overrides or ())
    invalid_tokens = [token for token in overrides if "=" not in token]
    if invalid_tokens:
        raise ScriptedBaselineConfigError(
            "Hydra overrides must use key=value syntax: " + ", ".join(invalid_tokens)
        )

    config = _compose_config(overrides)
    run_node = dict(config.get("run") or {})
    controller_node = dict(config.get("controller") or {})

    for key, value in (cli_overrides or {}).items():
        if key not in RUN_FIELD_NAMES:
            raise ScriptedBaselineConfigError(f"unknown legacy run override: {key}")
        if value is not None:
            run_node[key] = value

    return ScriptedBaselineCfg(
        run=_build_run_cfg(run_node),
        controller_overrides=_build_controller_overrides(controller_node),
    )


def apply_controller_overrides(
    base_cfg: DoorPushControllerCfg, overrides: dict[str, Any]
) -> DoorPushControllerCfg:
    """Return ``base_cfg`` with checked controller override values applied."""

    typed: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in CONTROLLER_FIELD_TYPES:
            raise ScriptedBaselineConfigError(f"unknown controller override: {key}")
        current = getattr(base_cfg, key)
        if isinstance(current, int) and not isinstance(current, bool):
            typed[key] = _coerce_int(f"controller.overrides.{key}", value)
        elif isinstance(current, float):
            typed[key] = _coerce_float(f"controller.overrides.{key}", value)
        else:
            typed[key] = value
    return replace(base_cfg, **typed)


def _compose_config(overrides: list[str]) -> dict[str, Any]:
    if not CONFIG_DIR.is_dir():
        raise ScriptedBaselineConfigError(f"config directory not found: {CONFIG_DIR}")

    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as error:  # noqa: BLE001 - normalize Hydra/OmegaConf errors for the CLI.
        raise ScriptedBaselineConfigError(str(error)) from error

    if not isinstance(container, dict):
        raise ScriptedBaselineConfigError("Hydra config did not resolve to a mapping")
    return container


def _build_run_cfg(node: dict[str, Any]) -> ScriptedBaselineRunCfg:
    unknown = sorted(set(node) - RUN_FIELD_NAMES)
    if unknown:
        raise ScriptedBaselineConfigError(
            "unknown run config field(s): " + ", ".join(unknown)
        )

    episodes = _coerce_int("run.episodes", node.get("episodes", 5))
    randomized = _coerce_int("run.randomized", node.get("randomized", 0))
    if episodes < 0:
        raise ScriptedBaselineConfigError("run.episodes must be non-negative")
    if randomized < 0:
        raise ScriptedBaselineConfigError("run.randomized must be non-negative")

    max_ticks = _coerce_int("run.max_ticks", node.get("max_ticks", 600))
    if max_ticks <= 0:
        raise ScriptedBaselineConfigError("run.max_ticks must be positive")

    success_angle_deg = _coerce_float(
        "run.success_angle_deg", node.get("success_angle_deg", 45.0)
    )
    if not math.isfinite(success_angle_deg):
        raise ScriptedBaselineConfigError("run.success_angle_deg must be finite")

    door_yaw_deg = _coerce_float("run.door_yaw_deg", node.get("door_yaw_deg", 0.0))
    door_offset_x = _coerce_float("run.door_offset_x", node.get("door_offset_x", 0.0))
    door_offset_y = _coerce_float("run.door_offset_y", node.get("door_offset_y", 0.0))
    for name, value in (
        ("run.door_yaw_deg", door_yaw_deg),
        ("run.door_offset_x", door_offset_x),
        ("run.door_offset_y", door_offset_y),
    ):
        if not math.isfinite(value):
            raise ScriptedBaselineConfigError(f"{name} must be finite")

    return ScriptedBaselineRunCfg(
        episodes=episodes,
        randomized=randomized,
        seed=_coerce_int("run.seed", node.get("seed", 0)),
        experiment=_optional_str("run.experiment", node.get("experiment")),
        run_id=_optional_str("run.run_id", node.get("run_id")),
        success_angle_deg=success_angle_deg,
        max_ticks=max_ticks,
        video=_coerce_bool("run.video", node.get("video", False)),
        clean_shutdown=_coerce_bool("run.clean_shutdown", node.get("clean_shutdown", False)),
        export=_coerce_bool("run.export", node.get("export", True)),
        door_yaw_deg=door_yaw_deg,
        door_offset_x=door_offset_x,
        door_offset_y=door_offset_y,
        door_pose_id=_optional_str("run.door_pose_id", node.get("door_pose_id")),
    )


def _build_controller_overrides(node: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(node.get("overrides") or {})
    unknown = sorted(set(overrides) - set(CONTROLLER_FIELD_TYPES))
    if unknown:
        raise ScriptedBaselineConfigError(
            "unknown controller override field(s): " + ", ".join(unknown)
        )
    return {key: value for key, value in overrides.items() if value is not None}


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ScriptedBaselineConfigError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ScriptedBaselineConfigError(f"{name} must be an integer") from error
    if integer != value and not (isinstance(value, float) and value.is_integer()):
        raise ScriptedBaselineConfigError(f"{name} must be an integer")
    return integer


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ScriptedBaselineConfigError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ScriptedBaselineConfigError(f"{name} must be a number") from error


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ScriptedBaselineConfigError(f"{name} must be a boolean")


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ScriptedBaselineConfigError(f"{name} must be a string or null")


__all__ = [
    "CONFIG_DIR",
    "CONFIG_NAME",
    "ScriptedBaselineCfg",
    "ScriptedBaselineConfigError",
    "ScriptedBaselineRunCfg",
    "apply_controller_overrides",
    "load_scripted_baseline_config",
]
