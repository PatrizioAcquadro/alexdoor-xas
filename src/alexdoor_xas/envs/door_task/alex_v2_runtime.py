"""Pure helpers that bind validated Alex V2 calibration to runtime config.

This module deliberately does not import Isaac Lab. It is therefore usable in
pure tests while the executor calls it after the Isaac application has loaded
the dedicated Alex V2 articulation configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alexdoor_xas.assets.alex_v2_tool_frame import derive_right_gripper_tool_frame
from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration

ALEX_V2_PRIM_PATH = "/World/envs/env_.*/Alex"
ALEX_V2_LIMITATIONS = (
    "Alex V2 is fixed-base: no stepping, balancing, or whole-body recovery is executed.",
    "Only the six right-arm joints are driven by position-only differential IK; "
    "rotation action components remain recorded but are not actuated.",
    "The controlled EE is the collision-derived gripper support point and its "
    "point Jacobian, both pinned to the validated runtime manifest.",
    "Contact force is selected by exact door actor ID from PhysX raw GPU contacts; "
    "unfiltered net force is never accepted.",
    "The standard articulated forearm collision union is used without external hands.",
)


class AlexV2RuntimeContractError(RuntimeError):
    """Raised when calibrated V2 execution cannot be constructed exactly."""


def calibrated_joint_position_map(
    calibration: AlexV2DoorCalibration,
) -> dict[str, float]:
    """Return six ready joints plus a non-overlapping zero catch-all."""

    ready = {name: float(value) for name, value in calibration.ready_joint_pos.items()}
    exclude = "|".join(ready)
    ready[f"(?!(?:{exclude})$).*"] = 0.0
    return ready


def inject_alex_v2_runtime_cfg(
    env_cfg: Any,
    robot_cfg: Any,
    calibration: AlexV2DoorCalibration,
) -> Any:
    """Inject only the dedicated V2 asset and calibrated initial conditions.

    ``robot_cfg`` must be the result of
    :func:`load_alex_v2_articulation_cfg`; this helper never accepts a URDF path
    or manufactures actuators itself.  The loose annotations keep this helper
    independently testable with small structural fakes.
    """

    if robot_cfg is None or not callable(getattr(robot_cfg, "replace", None)):
        raise AlexV2RuntimeContractError(
            "Alex V2 runtime requires a dedicated replaceable articulation config"
        )
    configured_robot = robot_cfg.replace(prim_path=ALEX_V2_PRIM_PATH)
    init_state = getattr(configured_robot, "init_state", None)
    if init_state is None:
        raise AlexV2RuntimeContractError("Alex V2 articulation config has no init_state")

    base_pose = calibration.base_pose
    init_state.pos = _tuple(base_pose["position_m"], 3, "base_pose.position_m")
    init_state.rot = _tuple(base_pose["orientation_xyzw"], 4, "base_pose.orientation_xyzw")
    init_state.joint_pos = calibrated_joint_position_map(calibration)
    env_cfg.robot = configured_robot
    env_cfg.contact_force_threshold_n = float(calibration.controller["contact_force_threshold_n"])
    return env_cfg


def require_current_collision_tool_frame(
    manifest: Mapping[str, Any],
    calibration: AlexV2DoorCalibration,
) -> None:
    """Reject a tool transform not reproduced from the current collision union."""

    normal = calibration.tool_frame["contact_normal_link"]
    derived = derive_right_gripper_tool_frame(manifest, normal).to_dict()
    active_fields = (
        "parent_link",
        "translation_m",
        "orientation_xyzw",
        "contact_normal_link",
    )
    current = {field: derived[field] for field in active_fields}
    if current != dict(calibration.tool_frame):
        raise AlexV2RuntimeContractError(
            "calibrated tool frame differs from the current collision manifest"
        )


def _tuple(value: Any, size: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AlexV2RuntimeContractError(f"{label} must be numeric") from error
    if len(result) != size:
        raise AlexV2RuntimeContractError(f"{label} must have length {size}")
    return result


__all__ = [
    "ALEX_V2_PRIM_PATH",
    "ALEX_V2_LIMITATIONS",
    "AlexV2RuntimeContractError",
    "calibrated_joint_position_map",
    "inject_alex_v2_runtime_cfg",
    "require_current_collision_tool_frame",
]
