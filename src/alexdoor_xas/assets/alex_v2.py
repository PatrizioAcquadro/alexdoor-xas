"""DoorManipulation loader for the pinned static Alex V2 asset."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_candidate_pd import apply_production_right_arm_pd
from alexdoor_xas.assets.alex_v2_contract import (
    DOOR_NON_RIGHT_ARM_DAMPING_SCALE,
    DOOR_RIGHT_ARM_ACTUATOR_NAME,
    DOOR_RIGHT_ARM_PD_GAINS,
    RobotAssetRef,
    derive_fixed_base_door_manifest,
    validate_alex_v2_manifest,
)
from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest


@dataclass(frozen=True)
class DoorAlexV2Asset:
    """Static URDF plus the distinct fixed-base Door runtime manifest."""

    urdf_path: Path
    manifest: dict[str, Any]
    fingerprint: str
    profile: str
    asset_id: str


def _apply_production_non_right_arm_damping(cfg: Any) -> None:
    """Scale body, left-arm, and gripper damping after right-arm isolation."""
    if cfg.spawn.self_collision is not True:
        raise ValueError("Alex V2 Door config must keep URDF self-collision enabled")
    if cfg.spawn.articulation_props.enabled_self_collisions is not True:
        raise ValueError("Alex V2 Door articulation must keep self-collision enabled")
    if not isinstance(cfg.actuators, Mapping) or not cfg.actuators:
        raise TypeError("Alex V2 Door actuators must be a non-empty mapping")
    if DOOR_RIGHT_ARM_ACTUATOR_NAME not in cfg.actuators:
        raise ValueError("Alex V2 Door production right-arm actuator is missing")

    scaled_by_actuator: dict[str, dict[str, float]] = {}
    for actuator_name, actuator_cfg in cfg.actuators.items():
        if actuator_name == DOOR_RIGHT_ARM_ACTUATOR_NAME:
            continue
        damping = actuator_cfg.damping
        if not isinstance(damping, Mapping) or not damping:
            raise TypeError(
                f"actuator {actuator_name!r} damping must be a non-empty mapping"
            )
        scaled: dict[str, float] = {}
        for joint_expression, value in damping.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"actuator {actuator_name!r} damping for "
                    f"{joint_expression!r} must be numeric"
                ) from error
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"actuator {actuator_name!r} damping for {joint_expression!r} "
                    "must be finite and non-negative"
                )
            scaled[str(joint_expression)] = (
                numeric * DOOR_NON_RIGHT_ARM_DAMPING_SCALE
            )
        scaled_by_actuator[str(actuator_name)] = scaled

    for actuator_name, scaled in scaled_by_actuator.items():
        cfg.actuators[actuator_name].damping = scaled


def build_alex_v2_door_asset(
    *,
    asset_root: str | Path | None = None,
) -> tuple[DoorAlexV2Asset, RobotAssetRef]:
    """Build and validate Door's runtime contract from the pinned static URDF."""
    root = Path(asset_root) if asset_root is not None else paths.ALEX_V2_ASSET_ROOT
    urdf_path = root.expanduser().resolve() / "urdf" / "alex_v2.urdf"
    static_manifest = build_alex_v2_manifest(urdf_path)
    validate_alex_v2_manifest(static_manifest)
    runtime_manifest = derive_fixed_base_door_manifest(static_manifest)
    ref = validate_alex_v2_manifest(runtime_manifest)
    asset = DoorAlexV2Asset(
        urdf_path=urdf_path,
        manifest=runtime_manifest,
        fingerprint=ref.manifest_fingerprint,
        profile=str(static_manifest["profile"]),
        asset_id=ref.asset_id,
    )
    return asset, ref


def load_alex_v2_articulation_cfg(
    *,
    fix_base: bool = True,
    asset_root: str | Path | None = None,
):
    """Return the dedicated V2 ``ArticulationCfg`` after ``AppLauncher``.

    Keeping the shared Isaac Lab configuration at the asset boundary prevents
    callers from substituting unvalidated asset or actuator definitions.
    """
    if fix_base is not True:
        raise ValueError("DoorManipulation's Alex V2 loader is fixed-base only")
    from ihmc_alex_isaaclab.robots.alex_v2 import make_alex_v2_cfg

    asset, _ = build_alex_v2_door_asset(asset_root=asset_root)
    cfg = make_alex_v2_cfg(str(asset.urdf_path), fix_base=True, variant="standard")
    if Path(cfg.spawn.asset_path).resolve() != Path(asset.urdf_path).resolve():
        raise ValueError("Alex V2 config factory did not retain the static URDF path")
    apply_production_right_arm_pd(cfg, ordered_gains=DOOR_RIGHT_ARM_PD_GAINS)
    _apply_production_non_right_arm_damping(cfg)
    return cfg


__all__ = [
    "DoorAlexV2Asset",
    "build_alex_v2_door_asset",
    "load_alex_v2_articulation_cfg",
]
