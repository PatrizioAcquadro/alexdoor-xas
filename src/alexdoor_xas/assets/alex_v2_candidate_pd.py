"""Production isolation and candidate overrides for Alex V2 right-arm PD.

This module is deliberately pure Python.  It splits the shared ``arms``
actuator for the fingerprinted production gains, then lets calibration probes
override only those six joints without changing the left arm, either gripper,
the body actuators, URDF limits, or production manifest.
"""

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

PRODUCTION_RIGHT_ARM_PD_PROFILE = "production_right_arm_pd_v2"

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

# Exact final gains used by calibration profiles.  stable_4x_v1 is sourced
# directly from the production contract; the other profiles remain
# candidate-only and make one controlled change at a time.
RIGHT_ARM_PD_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "stable_4x_v1": {
        joint_name: {"stiffness": stiffness, "damping": damping}
        for joint_name, stiffness, damping in DOOR_RIGHT_ARM_PD_GAINS
    },
    "balanced_3_5x_v1": {
        "RIGHT_SHOULDER_Y": {"stiffness": 26.78, "damping": 70.0},
        "RIGHT_SHOULDER_X": {"stiffness": 26.78, "damping": 70.0},
        "RIGHT_SHOULDER_Z": {"stiffness": 23.76, "damping": 35.0},
        "RIGHT_ELBOW_Y": {"stiffness": 23.76, "damping": 35.0},
        "RIGHT_WRIST_Z": {"stiffness": 5.0, "damping": 8.75},
        "RIGHT_WRIST_X": {"stiffness": 5.0, "damping": 8.75},
    },
    "responsive_k125_v1": {
        "RIGHT_SHOULDER_Y": {"stiffness": 33.475, "damping": 80.0},
        "RIGHT_SHOULDER_X": {"stiffness": 33.475, "damping": 80.0},
        "RIGHT_SHOULDER_Z": {"stiffness": 29.7, "damping": 40.0},
        "RIGHT_ELBOW_Y": {"stiffness": 29.7, "damping": 40.0},
        "RIGHT_WRIST_Z": {"stiffness": 6.25, "damping": 10.0},
        "RIGHT_WRIST_X": {"stiffness": 6.25, "damping": 10.0},
    },
}


def candidate_right_arm_pd_profile_names() -> tuple[str, ...]:
    """Return deterministic CLI choices for the candidate profiles."""
    return tuple(RIGHT_ARM_PD_PROFILES)


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


def _validated_profile(profile_name: str) -> dict[str, dict[str, float]]:
    try:
        profile = RIGHT_ARM_PD_PROFILES[profile_name]
    except KeyError as error:
        choices = ", ".join(candidate_right_arm_pd_profile_names())
        raise ValueError(
            f"unknown right-arm PD candidate {profile_name!r}; choose one of: {choices}"
        ) from error
    if tuple(profile) != RIGHT_ARM_PD_JOINTS:
        raise ValueError(
            f"right-arm PD candidate {profile_name!r} must contain the exact six-joint order"
        )
    validated: dict[str, dict[str, float]] = {}
    for joint_name, gains in profile.items():
        if not isinstance(gains, Mapping) or set(gains) != {"stiffness", "damping"}:
            raise ValueError(
                f"right-arm PD candidate {profile_name!r} joint {joint_name!r} "
                "must contain exactly stiffness and damping"
            )
        validated[joint_name] = {
            "stiffness": _finite_number(
                gains["stiffness"], field=f"{joint_name} stiffness", positive=True
            ),
            "damping": _finite_number(
                gains["damping"], field=f"{joint_name} damping", positive=True
            ),
        }
    return validated


def _copy_source_values(
    source: Mapping[str, Any], *, field: str, positive: bool
) -> dict[str, float]:
    copied: dict[str, float] = {}
    for joint_name in RIGHT_ARM_PD_JOINTS:
        expression = _SOURCE_EXPRESSION_BY_RIGHT_JOINT[joint_name]
        if expression not in source:
            raise ValueError(
                f"arms actuator {field} is missing source expression {expression!r}"
            )
        copied[joint_name] = _finite_number(
            source[expression],
            field=f"{joint_name} {field}",
            positive=positive,
        )
    return copied


def _copy_optional_source_field(source: Any, *, field: str) -> Any:
    """Re-key a per-joint mapping while preserving scalar/None source values."""
    if isinstance(source, Mapping):
        return _copy_source_values(source, field=field, positive=False)
    if source is None:
        return None
    return _finite_number(source, field=field, positive=False)


def _validated_ordered_gains(
    ordered_gains: tuple[tuple[str, float, float], ...],
) -> dict[str, dict[str, float]]:
    if tuple(item[0] for item in ordered_gains) != RIGHT_ARM_PD_JOINTS:
        raise ValueError("right-arm production gains must use the exact six-joint order")
    return {
        joint_name: {
            "stiffness": _finite_number(
                stiffness, field=f"{joint_name} stiffness", positive=True
            ),
            "damping": _finite_number(
                damping, field=f"{joint_name} damping", positive=True
            ),
        }
        for joint_name, stiffness, damping in ordered_gains
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


def _copy_exact_values(
    source: Any, *, field: str, positive: bool
) -> dict[str, float]:
    mapping = _numeric_mapping(source, field=field)
    if tuple(mapping) != RIGHT_ARM_PD_JOINTS:
        raise ValueError(f"production right-arm {field} must use the exact joint order")
    return {
        joint_name: _finite_number(
            mapping[joint_name], field=f"{joint_name} {field}", positive=positive
        )
        for joint_name in RIGHT_ARM_PD_JOINTS
    }


def apply_production_right_arm_pd(
    cfg: Any,
    *,
    ordered_gains: tuple[tuple[str, float, float], ...] = DOOR_RIGHT_ARM_PD_GAINS,
) -> dict[str, Any]:
    """Split the six production right-arm joints from the scaled arms actuator."""
    profile = _validated_ordered_gains(ordered_gains)
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
        _numeric_mapping(
            getattr(source, "velocity_limit_sim", None), field="velocity_limit_sim"
        ),
        field="velocity_limit_sim",
        positive=True,
    )
    effort = _copy_source_values(
        _numeric_mapping(
            getattr(source, "effort_limit_sim", None), field="effort_limit_sim"
        ),
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
    candidate = deepcopy(source)
    candidate.joint_names_expr = list(RIGHT_ARM_PD_JOINTS)
    candidate.stiffness = {
        name: profile[name]["stiffness"] for name in RIGHT_ARM_PD_JOINTS
    }
    candidate.damping = {name: profile[name]["damping"] for name in RIGHT_ARM_PD_JOINTS}
    candidate.velocity_limit_sim = velocity
    candidate.effort_limit_sim = effort
    candidate.armature = armature
    for field in (
        "effort_limit",
        "velocity_limit",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    ):
        setattr(
            candidate,
            field,
            _copy_optional_source_field(getattr(source, field, None), field=field),
        )

    actuators["arms"] = retained
    actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME] = candidate

    gains = {
        name: {
            **profile[name],
            "velocity_limit_sim": velocity[name],
            "effort_limit_sim": effort[name],
            "armature": armature[name],
        }
        for name in RIGHT_ARM_PD_JOINTS
    }
    return {
        "scope": "production_door_v2",
        "source_actuator": "arms",
        "right_arm_actuator": DOOR_RIGHT_ARM_ACTUATOR_NAME,
        "joint_order": list(RIGHT_ARM_PD_JOINTS),
        "gains": gains,
        "retained_arm_joints": list(_RETAINED_ARM_JOINTS),
        "right_arm_only": True,
        "position_limits": {"source": "URDF", "modified": False},
        "self_collision": {
            "urdf_import": True,
            "articulation": True,
        },
    }


def apply_candidate_right_arm_pd(cfg: Any, *, profile_name: str) -> dict[str, Any]:
    """Override the isolated production right arm for a calibration probe only."""
    profile = _validated_profile(profile_name)
    actuators = _validated_actuators(cfg)
    if DOOR_RIGHT_ARM_ACTUATOR_NAME not in actuators:
        raise TypeError(
            f"candidate config must contain production actuator "
            f"{DOOR_RIGHT_ARM_ACTUATOR_NAME!r}"
        )
    source = actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME]
    if tuple(getattr(source, "joint_names_expr", ())) != RIGHT_ARM_PD_JOINTS:
        raise ValueError("production right-arm actuator joint order differs")
    velocity = _copy_exact_values(
        getattr(source, "velocity_limit_sim", None),
        field="velocity_limit_sim",
        positive=True,
    )
    effort = _copy_exact_values(
        getattr(source, "effort_limit_sim", None),
        field="effort_limit_sim",
        positive=True,
    )
    armature = _copy_exact_values(
        getattr(source, "armature", None), field="armature", positive=False
    )
    candidate = deepcopy(source)
    candidate.stiffness = {
        name: profile[name]["stiffness"] for name in RIGHT_ARM_PD_JOINTS
    }
    candidate.damping = {name: profile[name]["damping"] for name in RIGHT_ARM_PD_JOINTS}
    actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME] = candidate
    gains = {
        name: {
            **profile[name],
            "velocity_limit_sim": velocity[name],
            "effort_limit_sim": effort[name],
            "armature": armature[name],
        }
        for name in RIGHT_ARM_PD_JOINTS
    }
    return {
        "scope": "calibration_probe_only",
        "profile": profile_name,
        "source_actuator": DOOR_RIGHT_ARM_ACTUATOR_NAME,
        "candidate_actuator": DOOR_RIGHT_ARM_ACTUATOR_NAME,
        "overrides_production_right_arm": True,
        "joint_order": list(RIGHT_ARM_PD_JOINTS),
        "gains": gains,
        "right_arm_only": True,
        "position_limits": {"source": "URDF", "modified": False},
        "self_collision": {
            "urdf_import": True,
            "articulation": True,
        },
        "production_config_modified": False,
        "production_manifest_modified": False,
    }


def apply_right_arm_pd_profile_selection(
    cfg: Any, *, profile_name: str
) -> dict[str, Any]:
    """Resolve a probe profile and apply only explicit candidate overrides.

    The canonical V2 spelling and ``none`` retain the production IK40 actuator
    byte-for-byte. Candidate profile names remain explicit calibration-probe
    overrides.
    """

    production_spellings = ("none", PRODUCTION_RIGHT_ARM_PD_PROFILE)
    if profile_name in production_spellings:
        return {
            "requested_profile": profile_name,
            "effective_profile": PRODUCTION_RIGHT_ARM_PD_PROFILE,
            "uses_production_right_arm_pd": True,
            "candidate_override": None,
        }
    candidate = apply_candidate_right_arm_pd(cfg, profile_name=profile_name)
    return {
        "requested_profile": profile_name,
        "effective_profile": profile_name,
        "uses_production_right_arm_pd": False,
        "candidate_override": candidate,
    }


__all__ = [
    "PRODUCTION_RIGHT_ARM_PD_PROFILE",
    "RIGHT_ARM_PD_JOINTS",
    "RIGHT_ARM_PD_PROFILES",
    "apply_candidate_right_arm_pd",
    "apply_production_right_arm_pd",
    "apply_right_arm_pd_profile_selection",
    "candidate_right_arm_pd_profile_names",
]
