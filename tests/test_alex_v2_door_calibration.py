"""Pure tests for the minimal active Alex V2 door calibration."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alexdoor_xas.calibration.alex_v2_door import (
    CALIBRATION_SCHEMA,
    CalibrationError,
    load_alex_v2_door_calibration,
)


def _payload() -> dict:
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "task": "door_push_alex_v2",
        "base_pose": {
            "position_m": [-0.45, -0.38, 0.93],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        "ready_joint_pos": {
            "RIGHT_SHOULDER_Y": 0.3,
            "RIGHT_SHOULDER_X": 0.0,
            "RIGHT_SHOULDER_Z": 0.0,
            "RIGHT_ELBOW_Y": -0.8,
            "RIGHT_WRIST_Z": 0.0,
            "RIGHT_WRIST_X": 0.0,
        },
        "tool_frame": {
            "parent_link": "RIGHT_GRIPPER_Z_LINK",
            "translation_m": [0.11, 0.0, -0.06],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "contact_normal_link": [1.0, 0.0, 0.0],
        },
        "reach_shell_m": [0.2, 0.8],
        "controller": {
            "push_radius_frac": 0.35,
            "push_height_m": 0.15,
            "approach_standoff_m": 0.12,
            "align_standoff_m": 0.10,
            "pre_contact_clearance_m": 0.01,
            "contact_clearance_m": -0.002,
            "contact_approach_max_step_m": 0.005,
            "release_standoff_m": 0.3,
            "contact_force_threshold_n": 1.5,
        },
        "randomization_bounds": {
            "start_offset_low": [-0.02, -0.02, -0.02],
            "start_offset_high": [0.02, 0.02, 0.02],
            "push_radius_frac_range": [0.33, 0.37],
            "push_height_m_range": [0.12, 0.17],
        },
    }


def _write(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_minimal_active_calibration(tmp_path) -> None:
    path = tmp_path / "alex_v2_door.json"
    _write(path, _payload())

    loaded = load_alex_v2_door_calibration(path)

    assert loaded.reach_shell_m == pytest.approx((0.2, 0.8))
    assert loaded.tool_frame["parent_link"] == "RIGHT_GRIPPER_Z_LINK"
    assert set(loaded.payload) == {
        "schema_version",
        "task",
        "base_pose",
        "ready_joint_pos",
        "tool_frame",
        "reach_shell_m",
        "controller",
        "randomization_bounds",
    }


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("task",), "wrong", "task must be"),
        (("tool_frame", "orientation_xyzw"), [0.0, 0.0, 0.0, 2.0], "unit length"),
        (("ready_joint_pos", "RIGHT_ELBOW_Y"), 1.0, "outside"),
        (("reach_shell_m",), [0.8, 0.2], "increasing"),
        (("controller", "contact_force_threshold_n"), float("nan"), "finite"),
        (("randomization_bounds", "push_radius_frac_range"), [0.4, 0.5], "excludes"),
    ],
)
def test_rejects_invalid_active_values(tmp_path, path, value, match) -> None:
    payload = deepcopy(_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    source = tmp_path / "calibration.json"
    _write(source, payload)

    with pytest.raises(CalibrationError, match=match):
        load_alex_v2_door_calibration(source)


def test_rejects_removed_administrative_fields_and_missing_file(tmp_path) -> None:
    payload = _payload()
    payload["status"] = "validated"
    source = tmp_path / "calibration.json"
    _write(source, payload)
    with pytest.raises(CalibrationError, match="unexpected status"):
        load_alex_v2_door_calibration(source)

    with pytest.raises(CalibrationError, match="unavailable"):
        load_alex_v2_door_calibration(tmp_path / "missing.json")
