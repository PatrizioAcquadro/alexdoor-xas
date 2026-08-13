"""Validated configuration for the scripted baseline CLI."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.door_task import DEFAULT_DOOR_POSE_ID, canonical_door_pose
from alexdoor_xas.policies.common.config import load_config
from alexdoor_xas.policies.scripted.door_push import DoorPushControllerCfg


class ScriptedBaselineConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ScriptedBaselineRunCfg:
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
    door_pose_id: str = DEFAULT_DOOR_POSE_ID


@dataclass(frozen=True)
class _ControllerNode:
    overrides: dict[str, Any]


@dataclass(frozen=True)
class _ConfigFile:
    run: ScriptedBaselineRunCfg
    controller: _ControllerNode


@dataclass(frozen=True)
class ScriptedBaselineCfg:
    run: ScriptedBaselineRunCfg
    controller_overrides: dict[str, Any]


def load_scripted_baseline_config(
    hydra_overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ScriptedBaselineCfg:
    dotted_cli = {f"run.{key}": value for key, value in (cli_overrides or {}).items()}
    config = load_config(
        paths.REPO_ROOT / "configs" / "scripted_baseline.yaml",
        _ConfigFile,
        ScriptedBaselineConfigError,
        hydra_overrides,
        dotted_cli,
    )
    _validate_run(config.run)
    known = {field.name for field in fields(DoorPushControllerCfg)}
    overrides = {
        key: value for key, value in config.controller.overrides.items() if value is not None
    }
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ScriptedBaselineConfigError(
            "unknown controller override field(s): " + ", ".join(unknown)
        )
    return ScriptedBaselineCfg(config.run, overrides)


def apply_controller_overrides(
    base_cfg: DoorPushControllerCfg, overrides: dict[str, Any]
) -> DoorPushControllerCfg:
    typed = {}
    for key, value in overrides.items():
        if not hasattr(base_cfg, key):
            raise ScriptedBaselineConfigError(f"unknown controller override: {key}")
        current = getattr(base_cfg, key)
        if isinstance(current, int) and not isinstance(current, bool):
            if isinstance(value, bool) or int(value) != value:
                raise ScriptedBaselineConfigError(f"controller.overrides.{key} must be an integer")
            value = int(value)
        elif isinstance(current, float):
            if isinstance(value, bool):
                raise ScriptedBaselineConfigError(f"controller.overrides.{key} must be a number")
            value = float(value)
        typed[key] = value
    return replace(base_cfg, **typed)


def _validate_run(run: ScriptedBaselineRunCfg) -> None:
    if run.episodes < 0:
        raise ScriptedBaselineConfigError("run.episodes must be non-negative")
    if run.randomized < 0:
        raise ScriptedBaselineConfigError("run.randomized must be non-negative")
    if run.max_ticks <= 0:
        raise ScriptedBaselineConfigError("run.max_ticks must be positive")
    if not math.isfinite(run.success_angle_deg):
        raise ScriptedBaselineConfigError("run.success_angle_deg must be finite")
    try:
        canonical_door_pose(run.door_pose_id)
    except ValueError as error:
        raise ScriptedBaselineConfigError(str(error)) from error
