"""Pure authoring helpers for the live Alex V2 door calibration gate."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from alexdoor_xas.assets.alex_v2_tool_frame import derive_right_gripper_tool_frame
from alexdoor_xas.calibration.alex_v2_door import (
    CALIBRATION_SCHEMA,
    RIGHT_ARM_JOINT_LIMITS_RAD,
    TASK_NAME,
    load_alex_v2_door_calibration,
)

REQUIRED_GATES = (
    "exact_runtime_joint_order",
    "reset_stability",
    "finite_jacobians",
    "collision_tool_frame",
    "contact_behavior",
    "fixed_scripted_baseline",
    "randomized_scripted_baseline",
)

BASE_POSE = {
    "position_m": [-0.45, -0.38, 0.93],
    "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
}
READY_JOINT_POS = {
    "RIGHT_SHOULDER_Y": 0.3,
    "RIGHT_SHOULDER_X": 0.0,
    "RIGHT_SHOULDER_Z": 0.0,
    "RIGHT_ELBOW_Y": -0.8,
    "RIGHT_WRIST_Z": 0.0,
    "RIGHT_WRIST_X": 0.0,
}
REACH_SHELL_M = (0.2, 0.8)
CONTROLLER = {
    "push_radius_frac": 0.35,
    "push_height_m": 0.15,
    "approach_standoff_m": 0.12,
    # Live seeds 0/1 proved the 0.06 m tool-point standoff still overlapped a
    # different gripper collision shape during ALIGN. Stay outside the measured
    # 0.08-0.09 m contact onset and enter contact only through PRE_CONTACT.
    "align_standoff_m": 0.10,
    "pre_contact_clearance_m": 0.01,
    "contact_clearance_m": -0.002,
    "contact_approach_max_step_m": 0.005,
    "release_standoff_m": 0.3,
    "contact_force_threshold_n": 1.5,
}
RANDOMIZATION_BOUNDS = {
    "start_offset_low": [-0.02, -0.02, -0.02],
    "start_offset_high": [0.02, 0.02, 0.02],
    "push_radius_frac_range": [0.33, 0.37],
    "push_height_m_range": [0.12, 0.17],
}


class CalibrationAuthoringError(RuntimeError):
    """Raised when live evidence is insufficient to author calibration."""


def compose_calibration_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compose the minimal active payload from the current runtime manifest."""

    _require_ready_pose_within_limits()
    derived = derive_right_gripper_tool_frame(manifest, (1.0, 0.0, 0.0)).to_dict()
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "task": TASK_NAME,
        "tool_frame": {
            field: deepcopy(derived[field])
            for field in (
                "parent_link",
                "translation_m",
                "orientation_xyzw",
                "contact_normal_link",
            )
        },
        "base_pose": deepcopy(BASE_POSE),
        "ready_joint_pos": dict(READY_JOINT_POS),
        "reach_shell_m": list(REACH_SHELL_M),
        "controller": dict(CONTROLLER),
        "randomization_bounds": deepcopy(RANDOMIZATION_BOUNDS),
    }


def all_required_gates_pass(gates: Mapping[str, Any]) -> bool:
    """Return true only for the exact required gate set with boolean true values."""

    return set(gates) == set(REQUIRED_GATES) and all(
        gates.get(name) is True for name in REQUIRED_GATES
    )


def write_calibration(
    destination: str | Path,
    payload: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate live gates, then atomically write the minimal calibration."""

    if not all_required_gates_pass(gates):
        failed = [name for name in REQUIRED_GATES if gates.get(name) is not True]
        raise CalibrationAuthoringError(
            "refusing to write production calibration; failed gates: " + ", ".join(failed)
        )
    calibration = deepcopy(dict(payload))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(calibration, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_alex_v2_door_calibration(temporary_name)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return calibration


def distance_envelope(values: Iterable[float]) -> tuple[float, float] | None:
    """Return the finite observed distance envelope, or ``None`` when invalid."""

    distances = [float(value) for value in values]
    if not distances or not all(math.isfinite(value) for value in distances):
        return None
    return min(distances), max(distances)


def envelope_within_shell(
    values: Iterable[float],
    shell_m: tuple[float, float] | list[float],
) -> bool:
    """Check that every observed shoulder-to-tool distance lies in the shell."""

    try:
        low, high = (float(value) for value in shell_m)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return False
    envelope = distance_envelope(values)
    return envelope is not None and low <= envelope[0] <= envelope[1] <= high


def make_reset_stability_evidence(
    *,
    finite_state: bool,
    grace_peak_abs_joint_velocity_rad_s: float,
    measured_peak_abs_joint_velocity_rad_s: float,
    grace_steps: int,
    measured_steps: int,
    bound_rad_s: float,
) -> dict[str, Any]:
    """Summarize a grace period followed by the bounded post-reset window."""

    if isinstance(grace_steps, bool) or not isinstance(grace_steps, int) or grace_steps < 0:
        raise ValueError("grace_steps must be a non-negative integer")
    if (
        isinstance(measured_steps, bool)
        or not isinstance(measured_steps, int)
        or measured_steps < 1
    ):
        raise ValueError("measured_steps must be a positive integer")
    grace_peak = float(grace_peak_abs_joint_velocity_rad_s)
    measured_peak = float(measured_peak_abs_joint_velocity_rad_s)
    bound = float(bound_rad_s)
    if not all(math.isfinite(value) and value >= 0.0 for value in (grace_peak, measured_peak)):
        raise ValueError("reset velocity peaks must be finite and non-negative")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("reset velocity bound must be finite and positive")
    return {
        "finite_state": bool(finite_state),
        "grace_peak_abs_joint_velocity_rad_s": grace_peak,
        "peak_abs_joint_velocity_rad_s": measured_peak,
        "bound_rad_s": bound,
        "grace_steps": grace_steps,
        "measured_steps": measured_steps,
        "total_steps": grace_steps + measured_steps,
        "passed": bool(finite_state and measured_peak < bound),
    }


def _require_ready_pose_within_limits() -> None:
    if set(READY_JOINT_POS) != set(RIGHT_ARM_JOINT_LIMITS_RAD):
        raise CalibrationAuthoringError("candidate ready pose must define exactly six arm joints")
    for name, value in READY_JOINT_POS.items():
        low, high = RIGHT_ARM_JOINT_LIMITS_RAD[name]
        if not low <= value <= high:
            raise CalibrationAuthoringError(
                f"candidate ready pose {name}={value} is outside [{low}, {high}] rad"
            )


__all__ = [
    "BASE_POSE",
    "CONTROLLER",
    "RANDOMIZATION_BOUNDS",
    "REACH_SHELL_M",
    "READY_JOINT_POS",
    "CalibrationAuthoringError",
    "all_required_gates_pass",
    "REQUIRED_GATES",
    "compose_calibration_payload",
    "distance_envelope",
    "envelope_within_shell",
    "make_reset_stability_evidence",
    "write_calibration",
]
