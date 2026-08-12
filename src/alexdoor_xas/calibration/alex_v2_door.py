"""Runtime contract for the Alex V2 door calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

_SCHEMA = "alexdoor.alex_v2_door.v1"
_TOOL_LINK = "RIGHT_GRIPPER_Z_LINK"
_MAX_CONTACT_STEP_M = 0.015
_UNIT_TOLERANCE = 1.0e-6

_ARM_JOINT_LIMITS_RAD = {
    "RIGHT_SHOULDER_Y": (-3.141592, 1.22173),
    "RIGHT_SHOULDER_X": (-2.79253, 0.349066),
    "RIGHT_SHOULDER_Z": (-1.22173, 1.91986),
    "RIGHT_ELBOW_Y": (-2.35619, 0.174532925),
    "RIGHT_WRIST_Z": (-2.61799, 2.61799),
    "RIGHT_WRIST_X": (-0.610865, 1.8326),
}
_ROOT_FIELDS = (
    "schema_version",
    "task",
    "base_pose",
    "ready_joint_pos",
    "tool_frame",
    "reach_shell_m",
    "controller",
    "randomization_bounds",
)
_TOOL_FRAME_FIELDS = (
    "parent_link",
    "translation_m",
    "orientation_xyzw",
    "contact_normal_link",
)
_CONTROLLER_FIELDS = (
    "push_radius_frac",
    "push_height_m",
    "approach_standoff_m",
    "align_standoff_m",
    "pre_contact_clearance_m",
    "contact_clearance_m",
    "contact_approach_max_step_m",
    "release_standoff_m",
    "contact_force_threshold_n",
)
_RANDOMIZATION_FIELDS = (
    "start_offset_low",
    "start_offset_high",
    "push_radius_frac_range",
    "push_height_m_range",
)


class CalibrationError(ValueError):
    """Invalid Alex V2 door calibration."""


@dataclass(frozen=True)
class AlexV2DoorCalibration:
    base_pose: Mapping[str, Any]
    ready_joint_pos: Mapping[str, float]
    tool_frame: Mapping[str, Any]
    reach_shell_m: tuple[float, float]
    controller: Mapping[str, float]
    randomization_bounds: Mapping[str, Any]


def load_alex_v2_door_calibration(path: str | Path) -> AlexV2DoorCalibration:
    calibration_path = Path(path).expanduser()
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"cannot load Alex V2 door calibration {calibration_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CalibrationError("Alex V2 door calibration must be a JSON object")
    return _validate_payload(payload)


def _validate_payload(payload: Mapping[str, Any]) -> AlexV2DoorCalibration:
    _require_exact_fields(payload, _ROOT_FIELDS, "calibration")
    if payload["schema_version"] != _SCHEMA:
        raise CalibrationError("Alex V2 door calibration schema is unsupported")
    if payload["task"] != paths.ALEX_V2_TASK:
        raise CalibrationError(f"Alex V2 door calibration task must be {paths.ALEX_V2_TASK!r}")
    controller = _validate_controller(payload["controller"])
    return AlexV2DoorCalibration(
        base_pose=_validate_pose(payload["base_pose"], "base_pose"),
        ready_joint_pos=_validate_ready_joint_pos(payload["ready_joint_pos"]),
        tool_frame=_validate_tool_frame(payload["tool_frame"]),
        reach_shell_m=_validate_reach_shell(payload["reach_shell_m"]),
        controller=controller,
        randomization_bounds=_validate_randomization(
            payload["randomization_bounds"], controller
        ),
    )


def _validate_tool_frame(value: Any) -> dict[str, Any]:
    frame = _mapping(value, "tool_frame")
    _require_exact_fields(frame, _TOOL_FRAME_FIELDS, "tool_frame")
    if frame["parent_link"] != _TOOL_LINK:
        raise CalibrationError(f"tool_frame.parent_link must be {_TOOL_LINK}")
    return {
        "parent_link": _TOOL_LINK,
        "translation_m": list(
            _finite_vector(frame["translation_m"], 3, "tool_frame.translation_m")
        ),
        "orientation_xyzw": list(
            _unit_vector(frame["orientation_xyzw"], 4, "tool_frame.orientation_xyzw")
        ),
        "contact_normal_link": list(
            _unit_vector(frame["contact_normal_link"], 3, "tool_frame.contact_normal_link")
        ),
    }


def _validate_pose(value: Any, label: str) -> dict[str, list[float]]:
    pose = _mapping(value, label)
    _require_exact_fields(pose, ("position_m", "orientation_xyzw"), label)
    return {
        "position_m": list(_finite_vector(pose["position_m"], 3, f"{label}.position_m")),
        "orientation_xyzw": list(
            _unit_vector(pose["orientation_xyzw"], 4, f"{label}.orientation_xyzw")
        ),
    }


def _validate_ready_joint_pos(value: Any) -> dict[str, float]:
    ready = _mapping(value, "ready_joint_pos")
    _require_exact_fields(ready, tuple(_ARM_JOINT_LIMITS_RAD), "ready_joint_pos")
    result: dict[str, float] = {}
    for name, (lower, upper) in _ARM_JOINT_LIMITS_RAD.items():
        position = _finite_number(ready[name], f"ready_joint_pos.{name}")
        if not lower <= position <= upper:
            raise CalibrationError(
                f"ready_joint_pos.{name}={position} is outside [{lower}, {upper}] rad"
            )
        result[name] = position
    return result


def _validate_reach_shell(value: Any) -> tuple[float, float]:
    low, high = _finite_vector(value, 2, "reach_shell_m")
    if not 0.0 < low < high:
        raise CalibrationError("reach_shell_m must be positive and increasing")
    return low, high


def _validate_controller(value: Any) -> dict[str, float]:
    controller = _mapping(value, "controller")
    _require_exact_fields(controller, _CONTROLLER_FIELDS, "controller")
    result = {
        name: _finite_number(controller[name], f"controller.{name}")
        for name in _CONTROLLER_FIELDS
    }
    radius = result["push_radius_frac"]
    approach = result["approach_standoff_m"]
    align = result["align_standoff_m"]
    pre_contact = result["pre_contact_clearance_m"]
    contact = result["contact_clearance_m"]
    contact_step = result["contact_approach_max_step_m"]
    release = result["release_standoff_m"]

    if not 0.0 < radius <= 1.0:
        raise CalibrationError("controller.push_radius_frac must be in (0, 1]")
    if approach <= 0.0:
        raise CalibrationError("controller.approach_standoff_m must be positive")
    if not 0.0 < align <= approach:
        raise CalibrationError(
            "controller.align_standoff_m must be positive and no greater than approach_standoff_m"
        )
    if not 0.0 <= pre_contact <= align:
        raise CalibrationError(
            "controller.pre_contact_clearance_m must be between zero and align_standoff_m"
        )
    if contact > pre_contact:
        raise CalibrationError(
            "controller.contact_clearance_m must not exceed pre_contact_clearance_m"
        )
    if not 0.0 < contact_step <= _MAX_CONTACT_STEP_M:
        raise CalibrationError(
            f"controller.contact_approach_max_step_m must be in (0, {_MAX_CONTACT_STEP_M}]"
        )
    if release < align:
        raise CalibrationError(
            "controller.release_standoff_m must not be less than align_standoff_m"
        )
    if result["contact_force_threshold_n"] <= 0.0:
        raise CalibrationError("controller.contact_force_threshold_n must be positive")
    return result


def _validate_randomization(
    value: Any, controller: Mapping[str, float]
) -> dict[str, list[float]]:
    bounds = _mapping(value, "randomization_bounds")
    _require_exact_fields(bounds, _RANDOMIZATION_FIELDS, "randomization_bounds")
    offset_low = _finite_vector(
        bounds["start_offset_low"], 3, "randomization_bounds.start_offset_low"
    )
    offset_high = _finite_vector(
        bounds["start_offset_high"], 3, "randomization_bounds.start_offset_high"
    )
    for axis, (low, high) in enumerate(zip(offset_low, offset_high, strict=True)):
        if not low <= 0.0 <= high:
            raise CalibrationError(
                f"randomization_bounds start offset axis {axis} must include zero"
            )

    radius_range = _finite_vector(
        bounds["push_radius_frac_range"], 2, "randomization_bounds.push_radius_frac_range"
    )
    height_range = _finite_vector(
        bounds["push_height_m_range"], 2, "randomization_bounds.push_height_m_range"
    )
    if not 0.0 < radius_range[0] <= radius_range[1] <= 1.0:
        raise CalibrationError("randomization_bounds.push_radius_frac_range must be inside (0, 1]")
    if height_range[0] > height_range[1]:
        raise CalibrationError("randomization_bounds.push_height_m_range must be increasing")

    _require_nominal_in_range(
        controller["push_radius_frac"],
        radius_range,
        "push_radius_frac_range",
        "push_radius_frac",
    )
    _require_nominal_in_range(
        controller["push_height_m"],
        height_range,
        "push_height_m_range",
        "push_height_m",
    )
    return {
        "start_offset_low": list(offset_low),
        "start_offset_high": list(offset_high),
        "push_radius_frac_range": list(radius_range),
        "push_height_m_range": list(height_range),
    }


def _require_nominal_in_range(
    nominal: float,
    bounds: tuple[float, float],
    bounds_name: str,
    controller_name: str,
) -> None:
    if not bounds[0] <= nominal <= bounds[1]:
        raise CalibrationError(
            f"randomization_bounds.{bounds_name} excludes controller.{controller_name}"
        )


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise CalibrationError(f"{label} must contain {size} numbers")
    return tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _unit_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    result = _finite_vector(value, size, label)
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_TOLERANCE):
        raise CalibrationError(f"{label} must be unit length")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{label} must be finite")
    return result


def _require_exact_fields(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise CalibrationError(f"{label} fields must match exactly: " + "; ".join(details))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{label} must be a mapping")
    return value


__all__ = ["AlexV2DoorCalibration", "CalibrationError", "load_alex_v2_door_calibration"]
