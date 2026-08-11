"""Active calibration contract for the ``door_push_alex_v2`` task."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

CALIBRATION_SCHEMA = "alexdoor.alex_v2_door.v1"
TASK_NAME = "door_push_alex_v2"
REQUIRED_CONTROLLER_FIELDS = (
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
REQUIRED_RANDOMIZATION_FIELDS = (
    "start_offset_low",
    "start_offset_high",
    "push_radius_frac_range",
    "push_height_m_range",
)
RIGHT_ARM_JOINT_LIMITS_RAD = {
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
_UNIT_TOLERANCE = 1.0e-6


class CalibrationError(ValueError):
    """Raised when the active calibration is missing or malformed."""


@dataclass(frozen=True)
class AlexV2DoorCalibration:
    path: Path
    payload: dict[str, Any]

    @property
    def tool_frame(self) -> Mapping[str, Any]:
        return self.payload["tool_frame"]

    @property
    def controller(self) -> Mapping[str, Any]:
        return self.payload["controller"]

    @property
    def randomization_bounds(self) -> Mapping[str, Any]:
        return self.payload["randomization_bounds"]

    @property
    def base_pose(self) -> Mapping[str, Any]:
        return self.payload["base_pose"]

    @property
    def ready_joint_pos(self) -> Mapping[str, float]:
        return self.payload["ready_joint_pos"]

    @property
    def reach_shell_m(self) -> tuple[float, float]:
        low, high = self.payload["reach_shell_m"]
        return float(low), float(high)


def load_alex_v2_door_calibration(
    path: str | Path | None = None,
) -> AlexV2DoorCalibration:
    """Load and validate the single active Alex V2 door calibration."""

    calibration_path = default_calibration_path() if path is None else Path(path).expanduser()
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"Alex V2 door calibration is unavailable: {calibration_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CalibrationError("Alex V2 door calibration must be a JSON object")
    _validate_payload(payload)
    return AlexV2DoorCalibration(calibration_path, payload)


def default_calibration_path() -> Path:
    return paths.REPO_ROOT / "configs" / "alex_v2_door.json"


def _validate_payload(payload: Mapping[str, Any]) -> None:
    _require_exact_fields(payload, _ROOT_FIELDS, "calibration")
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise CalibrationError("Alex V2 door calibration schema is unsupported")
    if payload.get("task") != TASK_NAME:
        raise CalibrationError(f"Alex V2 door calibration task must be {TASK_NAME!r}")
    _validate_tool_frame(payload.get("tool_frame"))
    _validate_pose(payload.get("base_pose"), "base_pose")
    _validate_ready_joint_pos(payload.get("ready_joint_pos"))
    _validate_reach_shell_m(payload.get("reach_shell_m"))
    controller = _validate_controller(payload.get("controller"))
    _validate_randomization_bounds(payload.get("randomization_bounds"), controller)


def _validate_tool_frame(value: Any) -> None:
    frame = _mapping(value, "tool_frame")
    _require_exact_fields(frame, _TOOL_FRAME_FIELDS, "tool_frame")
    if frame.get("parent_link") != "RIGHT_GRIPPER_Z_LINK":
        raise CalibrationError("tool frame must be attached to RIGHT_GRIPPER_Z_LINK")
    _finite_vector(frame.get("translation_m"), 3, "tool_frame.translation_m")
    _unit_vector(frame.get("orientation_xyzw"), 4, "tool_frame.orientation_xyzw")
    _unit_vector(frame.get("contact_normal_link"), 3, "tool_frame.contact_normal_link")


def _validate_pose(value: Any, label: str) -> None:
    pose = _mapping(value, label)
    _require_exact_fields(pose, ("position_m", "orientation_xyzw"), label)
    _finite_vector(pose.get("position_m"), 3, f"{label}.position_m")
    _unit_vector(pose.get("orientation_xyzw"), 4, f"{label}.orientation_xyzw")


def _validate_ready_joint_pos(value: Any) -> None:
    ready = _mapping(value, "ready_joint_pos")
    _require_exact_fields(ready, tuple(RIGHT_ARM_JOINT_LIMITS_RAD), "ready_joint_pos")
    for name, (lower, upper) in RIGHT_ARM_JOINT_LIMITS_RAD.items():
        position = _finite_number(ready[name], f"ready_joint_pos.{name}")
        if not lower <= position <= upper:
            raise CalibrationError(
                f"ready_joint_pos.{name}={position} is outside [{lower}, {upper}] rad"
            )


def _validate_reach_shell_m(value: Any) -> None:
    low, high = _finite_vector(value, 2, "reach_shell_m")
    if low <= 0.0 or low >= high:
        raise CalibrationError("reach_shell_m must be a positive increasing interval")


def _validate_controller(value: Any) -> dict[str, float]:
    controller = _mapping(value, "controller")
    _require_exact_fields(controller, REQUIRED_CONTROLLER_FIELDS, "controller")
    values = {
        name: _finite_number(controller[name], f"controller.{name}")
        for name in REQUIRED_CONTROLLER_FIELDS
    }
    _require_range(
        values["push_radius_frac"], 0.0, 1.0, "controller.push_radius_frac", open_low=True
    )
    _require_range(values["push_height_m"], -1.0, 1.0, "controller.push_height_m")
    _require_range(
        values["approach_standoff_m"], 0.0, 1.0, "controller.approach_standoff_m", open_low=True
    )
    _require_range(
        values["align_standoff_m"],
        0.0,
        values["approach_standoff_m"],
        "controller.align_standoff_m",
        open_low=True,
    )
    _require_range(
        values["pre_contact_clearance_m"],
        0.0,
        values["align_standoff_m"],
        "controller.pre_contact_clearance_m",
    )
    _require_range(
        values["contact_clearance_m"],
        -0.05,
        values["pre_contact_clearance_m"],
        "controller.contact_clearance_m",
    )
    _require_range(
        values["contact_approach_max_step_m"],
        0.0,
        0.015,
        "controller.contact_approach_max_step_m",
        open_low=True,
    )
    _require_range(
        values["release_standoff_m"],
        values["align_standoff_m"],
        1.0,
        "controller.release_standoff_m",
    )
    _require_range(
        values["contact_force_threshold_n"],
        0.0,
        1000.0,
        "controller.contact_force_threshold_n",
        open_low=True,
    )
    return values


def _validate_randomization_bounds(value: Any, controller: Mapping[str, float]) -> None:
    bounds = _mapping(value, "randomization_bounds")
    _require_exact_fields(bounds, REQUIRED_RANDOMIZATION_FIELDS, "randomization_bounds")
    offset_low = _finite_vector(
        bounds["start_offset_low"], 3, "randomization_bounds.start_offset_low"
    )
    offset_high = _finite_vector(
        bounds["start_offset_high"], 3, "randomization_bounds.start_offset_high"
    )
    for axis, (low, high) in enumerate(zip(offset_low, offset_high, strict=True)):
        if low > high or not low <= 0.0 <= high:
            raise CalibrationError(
                f"randomization_bounds start offset axis {axis} must include zero"
            )
    for field, controller_field in (
        ("push_radius_frac_range", "push_radius_frac"),
        ("push_height_m_range", "push_height_m"),
    ):
        low, high = _finite_vector(bounds[field], 2, f"randomization_bounds.{field}")
        if low > high:
            raise CalibrationError(f"randomization_bounds.{field} must be increasing")
        if controller_field == "push_radius_frac":
            _require_range(low, 0.0, 1.0, f"randomization_bounds.{field}[0]", open_low=True)
            _require_range(high, 0.0, 1.0, f"randomization_bounds.{field}[1]", open_low=True)
        else:
            _require_range(low, -1.0, 1.0, f"randomization_bounds.{field}[0]")
            _require_range(high, -1.0, 1.0, f"randomization_bounds.{field}[1]")
        if not low <= controller[controller_field] <= high:
            raise CalibrationError(
                f"randomization_bounds.{field} excludes controller.{controller_field}"
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


def _require_range(
    value: float, low: float, high: float, label: str, *, open_low: bool = False
) -> None:
    if not (value > low if open_low else value >= low) or value > high:
        left = "(" if open_low else "["
        raise CalibrationError(f"{label} must be in {left}{low}, {high}]")


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


__all__ = [
    "CALIBRATION_SCHEMA",
    "RIGHT_ARM_JOINT_LIMITS_RAD",
    "TASK_NAME",
    "AlexV2DoorCalibration",
    "CalibrationError",
    "default_calibration_path",
    "load_alex_v2_door_calibration",
]
