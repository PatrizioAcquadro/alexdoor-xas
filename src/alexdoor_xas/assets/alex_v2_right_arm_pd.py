"""Production Alex V2 right-arm PD isolation for the Door benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from alexdoor_xas.assets.alex_v2_contract import (
    DOOR_RIGHT_ARM_ACTUATOR_NAME,
    DOOR_RIGHT_ARM_PD_GAINS,
)

RIGHT_ARM_PD_JOINTS = (
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
)

_SOURCE_ARM_EXPRESSIONS = (
    ".*SHOULDER_Y",
    ".*SHOULDER_X",
    ".*SHOULDER_Z",
    ".*ELBOW_Y",
    ".*WRIST_Z",
    ".*WRIST_X",
    ".*GRIPPER_Z",
)
_SOURCE_EXPRESSION_BY_RIGHT_JOINT = {
    "RIGHT_SHOULDER_Y": ".*SHOULDER_Y",
    "RIGHT_SHOULDER_X": ".*SHOULDER_X",
    "RIGHT_SHOULDER_Z": ".*SHOULDER_Z",
    "RIGHT_ELBOW_Y": ".*ELBOW_Y",
    "RIGHT_WRIST_Z": ".*WRIST_Z",
    "RIGHT_WRIST_X": ".*WRIST_X",
}
_RETAINED_ARM_JOINTS = (
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "RIGHT_GRIPPER_Z",
)


def _numeric_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"actuator {field} must be a non-empty mapping")
    return value


def _finite_number(value: Any, *, field: str, positive: bool) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be numeric") from error
    if not math.isfinite(numeric) or (numeric <= 0.0 if positive else numeric < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return numeric


def _copy_source_values(
    source: Mapping[str, Any], *, field: str, positive: bool
) -> dict[str, float]:
    copied: dict[str, float] = {}
    for joint_name in RIGHT_ARM_PD_JOINTS:
        expression = _SOURCE_EXPRESSION_BY_RIGHT_JOINT[joint_name]
        if expression not in source:
            raise ValueError(f"arms actuator {field} is missing source expression {expression!r}")
        copied[joint_name] = _finite_number(
            source[expression],
            field=f"{joint_name} {field}",
            positive=positive,
        )
    return copied


def _copy_optional_source_field(source: Any, *, field: str) -> Any:
    """Re-key a per-joint mapping while preserving scalar or absent source values."""

    if isinstance(source, Mapping):
        return _copy_source_values(source, field=field, positive=False)
    if source is None:
        return None
    return _finite_number(source, field=field, positive=False)


def _validated_ordered_gains() -> dict[str, dict[str, float]]:
    if tuple(item[0] for item in DOOR_RIGHT_ARM_PD_GAINS) != RIGHT_ARM_PD_JOINTS:
        raise ValueError("right-arm production gains must use the exact six-joint order")
    return {
        joint_name: {
            "stiffness": _finite_number(stiffness, field=f"{joint_name} stiffness", positive=True),
            "damping": _finite_number(damping, field=f"{joint_name} damping", positive=True),
        }
        for joint_name, stiffness, damping in DOOR_RIGHT_ARM_PD_GAINS
    }


def _validated_actuators(cfg: Any) -> dict[str, Any]:
    spawn = getattr(cfg, "spawn", None)
    articulation_props = getattr(spawn, "articulation_props", None)
    if getattr(spawn, "self_collision", None) is not True:
        raise ValueError("right-arm PD requires URDF self-collision enabled")
    if getattr(articulation_props, "enabled_self_collisions", None) is not True:
        raise ValueError("right-arm PD requires articulation self-collision enabled")
    actuators = getattr(cfg, "actuators", None)
    if not isinstance(actuators, dict):
        raise TypeError("right-arm PD config must contain a mutable actuator mapping")
    return actuators


def apply_production_right_arm_pd(cfg: Any) -> None:
    """Split the six production right-arm joints from the shared arms actuator."""

    gains_profile = _validated_ordered_gains()
    actuators = _validated_actuators(cfg)
    if "arms" not in actuators:
        raise TypeError("production config must contain the shared 'arms' actuator")
    if DOOR_RIGHT_ARM_ACTUATOR_NAME in actuators:
        raise ValueError("production right-arm actuator was already applied")

    source = actuators["arms"]
    if tuple(getattr(source, "joint_names_expr", ())) != _SOURCE_ARM_EXPRESSIONS:
        raise ValueError(
            "arms actuator joint expressions differ from the dedicated Alex V2 source config"
        )
    _numeric_mapping(getattr(source, "stiffness", None), field="stiffness")
    _numeric_mapping(getattr(source, "damping", None), field="damping")
    velocity = _copy_source_values(
        _numeric_mapping(getattr(source, "velocity_limit_sim", None), field="velocity_limit_sim"),
        field="velocity_limit_sim",
        positive=True,
    )
    effort = _copy_source_values(
        _numeric_mapping(getattr(source, "effort_limit_sim", None), field="effort_limit_sim"),
        field="effort_limit_sim",
        positive=True,
    )
    armature = _copy_source_values(
        _numeric_mapping(getattr(source, "armature", None), field="armature"),
        field="armature",
        positive=False,
    )

    retained = deepcopy(source)
    retained.joint_names_expr = list(_RETAINED_ARM_JOINTS)
    right_arm = deepcopy(source)
    right_arm.joint_names_expr = list(RIGHT_ARM_PD_JOINTS)
    right_arm.stiffness = {name: gains_profile[name]["stiffness"] for name in RIGHT_ARM_PD_JOINTS}
    right_arm.damping = {name: gains_profile[name]["damping"] for name in RIGHT_ARM_PD_JOINTS}
    right_arm.velocity_limit_sim = velocity
    right_arm.effort_limit_sim = effort
    right_arm.armature = armature
    for field in (
        "effort_limit",
        "velocity_limit",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    ):
        setattr(
            right_arm,
            field,
            _copy_optional_source_field(getattr(source, field, None), field=field),
        )

    actuators["arms"] = retained
    actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME] = right_arm


__all__ = ["RIGHT_ARM_PD_JOINTS", "apply_production_right_arm_pd"]
