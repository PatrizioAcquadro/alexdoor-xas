"""Fail-closed calibration contract for the ``door_push_alex_v2`` task."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef

CALIBRATION_SCHEMA = "alexdoor.alex_v2_door_calibration.v0"
TASK_NAME = "door_push_alex_v2"
REQUIRED_GATES = (
    "exact_runtime_joint_order",
    "reset_stability",
    "finite_jacobians",
    "collision_tool_frame",
    "contact_behavior",
    "fixed_scripted_baseline",
    "randomized_scripted_baseline",
)
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

# Frozen from the pinned IHMC Alex SDK 0.4.0 standard V2 URDF fragments.
RIGHT_ARM_JOINT_LIMITS_RAD = {
    "RIGHT_SHOULDER_Y": (-3.141592, 1.22173),
    "RIGHT_SHOULDER_X": (-2.79253, 0.349066),
    "RIGHT_SHOULDER_Z": (-1.22173, 1.91986),
    "RIGHT_ELBOW_Y": (-2.35619, 0.174532925),
    "RIGHT_WRIST_Z": (-2.61799, 2.61799),
    "RIGHT_WRIST_X": (-0.610865, 1.8326),
}

_UNIT_TOLERANCE = 1.0e-6


class CalibrationError(ValueError):
    """Raised when a task attempts to use unvalidated or stale calibration."""


@dataclass(frozen=True)
class AlexV2DoorCalibration:
    path: Path
    payload: dict[str, Any]
    robot_asset: RobotAssetRef

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

    @property
    def status(self) -> str:
        return str(self.payload["status"])


def load_validated_alex_v2_door_calibration(
    path: str | Path,
    *,
    runtime_asset: RobotAssetRef,
    runtime_versions: Mapping[str, str],
) -> AlexV2DoorCalibration:
    """Load calibration only when all identity and live-evidence gates match."""

    return _load_alex_v2_door_calibration(
        path,
        runtime_asset=runtime_asset,
        runtime_versions=runtime_versions,
        expected_status="validated",
        require_complete_gates=True,
    )


def load_candidate_alex_v2_door_calibration(
    path: str | Path,
    *,
    runtime_asset: RobotAssetRef,
    runtime_versions: Mapping[str, str],
) -> AlexV2DoorCalibration:
    """Load a structurally complete candidate for the unregistered probe env.

    Candidate calibration is still pinned to the exact robot asset and runtime,
    fingerprinted, and validated field-by-field.  Its live gates may be false,
    but every required gate must be explicitly represented by a boolean.  The
    registered production task never calls this loader.
    """

    return _load_alex_v2_door_calibration(
        path,
        runtime_asset=runtime_asset,
        runtime_versions=runtime_versions,
        expected_status="candidate",
        require_complete_gates=False,
    )


def _load_alex_v2_door_calibration(
    path: str | Path,
    *,
    runtime_asset: RobotAssetRef,
    runtime_versions: Mapping[str, str],
    expected_status: str,
    require_complete_gates: bool,
) -> AlexV2DoorCalibration:
    """Shared parser; status selection stays explicit at the public boundary."""

    calibration_path = Path(path).expanduser()
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"validated Alex V2 door calibration is unavailable: {calibration_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CalibrationError("Alex V2 door calibration must be a JSON object")
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise CalibrationError(
            "Alex V2 door calibration schema is missing or unsupported"
        )
    status = payload.get("status")
    if status != expected_status:
        if expected_status == "validated":
            raise CalibrationError(
                "Alex V2 door calibration is not validated; "
                "candidate evidence cannot run the production task"
            )
        raise CalibrationError(
            "Alex V2 candidate calibration must have status='candidate'; "
            "validated evidence belongs to the production task"
        )
    if payload.get("task") != TASK_NAME:
        raise CalibrationError(
            f"Alex V2 door calibration task must be exactly {TASK_NAME!r}"
        )
    stored_asset = _robot_asset(payload.get("robot_asset"))
    if stored_asset != runtime_asset:
        raise CalibrationError(
            "Alex V2 door calibration belongs to a different robot asset"
        )
    stored_runtime = _mapping(payload.get("runtime_versions"), "runtime_versions")
    if dict(stored_runtime) != dict(runtime_versions):
        raise CalibrationError("Alex V2 door calibration runtime versions do not match")
    gates = _validate_gates(payload.get("gates"))
    failed = [name for name in REQUIRED_GATES if not gates[name]]
    if require_complete_gates and failed:
        raise CalibrationError(
            "Alex V2 door calibration gates are incomplete: " + ", ".join(failed)
        )
    _validate_tool_frame(payload.get("tool_frame"))
    _validate_pose(payload.get("base_pose"), "base_pose")
    _validate_ready_joint_pos(payload.get("ready_joint_pos"))
    _validate_reach_shell_m(payload.get("reach_shell_m"))
    controller = _validate_controller(payload.get("controller"))
    _validate_randomization_bounds(payload.get("randomization_bounds"), controller)
    expected_fingerprint = calibration_fingerprint(payload)
    if payload.get("fingerprint") != expected_fingerprint:
        raise CalibrationError("Alex V2 door calibration fingerprint mismatch")
    return AlexV2DoorCalibration(calibration_path, payload, stored_asset)


def calibration_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("fingerprint", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def default_calibration_path() -> Path:
    return paths.REPO_ROOT / "configs" / "alex_v2_door_calibration.v0.json"


def _robot_asset(value: Any) -> RobotAssetRef:
    mapping = _mapping(value, "robot_asset")
    try:
        return RobotAssetRef.from_dict(mapping)
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(
            f"invalid robot_asset calibration identity: {error}"
        ) from error


def _validate_tool_frame(value: Any) -> None:
    frame = _mapping(value, "tool_frame")
    if frame.get("parent_link") != "RIGHT_GRIPPER_Z_LINK":
        raise CalibrationError("tool frame must be attached to RIGHT_GRIPPER_Z_LINK")
    _finite_vector(frame.get("translation_m"), 3, "tool_frame.translation_m")
    _unit_vector(frame.get("orientation_xyzw"), 4, "tool_frame.orientation_xyzw")
    _unit_vector(frame.get("contact_normal_link"), 3, "tool_frame.contact_normal_link")
    fingerprint = str(frame.get("collision_union_sha256", ""))
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise CalibrationError("tool frame collision union fingerprint is invalid")
    support_distance = _finite_number(
        frame.get("support_distance_m"), "tool_frame.support_distance_m"
    )
    if support_distance <= 0.0:
        raise CalibrationError("tool_frame.support_distance_m must be positive")


def _validate_pose(value: Any, label: str) -> None:
    pose = _mapping(value, label)
    _finite_vector(pose.get("position_m"), 3, f"{label}.position_m")
    _unit_vector(pose.get("orientation_xyzw"), 4, f"{label}.orientation_xyzw")


def _validate_ready_joint_pos(value: Any) -> None:
    ready = _mapping(value, "ready_joint_pos")
    _require_exact_fields(ready, tuple(RIGHT_ARM_JOINT_LIMITS_RAD), "ready_joint_pos")
    for name, (lower, upper) in RIGHT_ARM_JOINT_LIMITS_RAD.items():
        position = _finite_number(ready[name], f"ready_joint_pos.{name}")
        if not lower <= position <= upper:
            raise CalibrationError(
                f"ready_joint_pos.{name}={position} is outside official V2 limits "
                f"[{lower}, {upper}] rad"
            )


def _validate_reach_shell_m(value: Any) -> None:
    low, high = _finite_vector(value, 2, "reach_shell_m")
    if low <= 0.0:
        raise CalibrationError("reach_shell_m minimum must be positive")
    if low >= high:
        raise CalibrationError("reach_shell_m minimum must be less than maximum")


def _validate_controller(value: Any) -> dict[str, float]:
    controller = _mapping(value, "controller")
    _require_exact_fields(controller, REQUIRED_CONTROLLER_FIELDS, "controller")
    values = {
        name: _finite_number(controller[name], f"controller.{name}")
        for name in REQUIRED_CONTROLLER_FIELDS
    }

    _require_range(
        values["push_radius_frac"],
        0.0,
        1.0,
        "controller.push_radius_frac",
        open_low=True,
    )
    _require_range(values["push_height_m"], -1.0, 1.0, "controller.push_height_m")
    _require_range(
        values["approach_standoff_m"],
        0.0,
        1.0,
        "controller.approach_standoff_m",
        open_low=True,
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


def _validate_randomization_bounds(
    value: Any,
    controller: Mapping[str, float],
) -> None:
    bounds = _mapping(value, "randomization_bounds")
    _require_exact_fields(bounds, REQUIRED_RANDOMIZATION_FIELDS, "randomization_bounds")

    offset_low = _finite_vector(
        bounds["start_offset_low"], 3, "randomization_bounds.start_offset_low"
    )
    offset_high = _finite_vector(
        bounds["start_offset_high"], 3, "randomization_bounds.start_offset_high"
    )
    for axis, (low, high) in enumerate(zip(offset_low, offset_high, strict=True)):
        if low > high:
            raise CalibrationError(
                f"randomization_bounds start offset axis {axis} has low greater than high"
            )
        if not low <= 0.0 <= high:
            raise CalibrationError(
                f"randomization_bounds start offset axis {axis} excludes nominal zero"
            )

    for field, controller_field in (
        ("push_radius_frac_range", "push_radius_frac"),
        ("push_height_m_range", "push_height_m"),
    ):
        low, high = _finite_vector(bounds[field], 2, f"randomization_bounds.{field}")
        if low > high:
            raise CalibrationError(
                f"randomization_bounds.{field} has low greater than high"
            )
        if controller_field == "push_radius_frac":
            _require_range(
                low,
                0.0,
                1.0,
                f"randomization_bounds.{field}[0]",
                open_low=True,
            )
            _require_range(
                high, 0.0, 1.0, f"randomization_bounds.{field}[1]", open_low=True
            )
        else:
            _require_range(low, -1.0, 1.0, f"randomization_bounds.{field}[0]")
            _require_range(high, -1.0, 1.0, f"randomization_bounds.{field}[1]")
        nominal = controller[controller_field]
        if not low <= nominal <= high:
            raise CalibrationError(
                f"randomization_bounds.{field} excludes nominal controller.{controller_field}"
            )


def _validate_gates(value: Any) -> dict[str, bool]:
    gates = _mapping(value, "gates")
    _require_exact_fields(gates, REQUIRED_GATES, "gates")
    result: dict[str, bool] = {}
    for name in REQUIRED_GATES:
        gate = gates[name]
        if not isinstance(gate, bool):
            raise CalibrationError(f"gates.{name} must be a boolean")
        result[name] = gate
    return result


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise CalibrationError(f"{label} must contain {size} numbers")
    return tuple(
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


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
    value: float,
    low: float,
    high: float,
    label: str,
    *,
    open_low: bool = False,
) -> None:
    low_ok = value > low if open_low else value >= low
    if not low_ok or value > high:
        left = "(" if open_low else "["
        raise CalibrationError(f"{label} must be in {left}{low}, {high}]")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: Sequence[str],
    label: str,
) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise CalibrationError(
            f"{label} fields must match exactly: " + "; ".join(details)
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{label} must be a mapping")
    return value


__all__ = [
    "CALIBRATION_SCHEMA",
    "REQUIRED_GATES",
    "RIGHT_ARM_JOINT_LIMITS_RAD",
    "TASK_NAME",
    "AlexV2DoorCalibration",
    "CalibrationError",
    "calibration_fingerprint",
    "default_calibration_path",
    "load_candidate_alex_v2_door_calibration",
    "load_validated_alex_v2_door_calibration",
]
