"""Pure loading and live-gated authoring contracts for Alex V2 calibration."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alexdoor_xas.calibration.alex_v2_door import (
    CALIBRATION_SCHEMA,
    CalibrationError,
    load_alex_v2_door_calibration,
)
from alexdoor_xas.calibration.alex_v2_door_authoring import (
    REQUIRED_GATES,
    CalibrationAuthoringError,
    compose_calibration_payload,
    distance_envelope,
    envelope_within_shell,
    make_reset_stability_evidence,
    write_calibration,
)

# --- test_alex_v2_door_calibration ---


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


# --- test_alex_v2_door_calibration_authoring ---


def _manifest() -> dict:
    return {
        "collision_profile": {
            "links": {
                "RIGHT_GRIPPER_Z_LINK": [
                    {
                        "name": "right_thumb_collision",
                        "link": "RIGHT_GRIPPER_Z_LINK",
                        "shape": "sphere",
                        "origin": {
                            "xyz_m": [0.1, 0.0, -0.06],
                            "rpy_rad": [0.0, 0.0, 0.0],
                        },
                        "dimensions": {"radius_m": 0.02},
                    }
                ]
            }
        }
    }


def test_composed_payload_contains_only_active_fields(tmp_path) -> None:
    payload = compose_calibration_payload(_manifest())
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload))

    loaded = load_alex_v2_door_calibration(path)

    assert loaded.tool_frame["translation_m"] == pytest.approx([0.12, 0.0, -0.06])
    assert "robot_asset" not in payload
    assert "runtime_versions" not in payload
    assert "gates" not in payload
    assert "fingerprint" not in payload


@pytest.mark.parametrize("failed_gate", REQUIRED_GATES)
def test_writer_refuses_each_failed_live_gate(tmp_path, failed_gate) -> None:
    destination = tmp_path / "configs" / "alex_v2_door.json"
    gates = {name: True for name in REQUIRED_GATES}
    gates[failed_gate] = False

    with pytest.raises(CalibrationAuthoringError, match=failed_gate):
        write_calibration(destination, compose_calibration_payload(_manifest()), gates)

    assert not destination.exists()


def test_writer_authors_loadable_payload_atomically(tmp_path) -> None:
    destination = tmp_path / "configs" / "alex_v2_door.json"
    gates = {name: True for name in REQUIRED_GATES}

    written = write_calibration(destination, compose_calibration_payload(_manifest()), gates)
    loaded = load_alex_v2_door_calibration(destination)

    assert written == loaded.payload
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_distance_envelope_and_reset_stability() -> None:
    assert distance_envelope([0.4, 0.2, 0.7]) == pytest.approx((0.2, 0.7))
    assert envelope_within_shell([0.2, 0.4, 0.8], (0.2, 0.8))
    assert not envelope_within_shell([0.19, 0.4], (0.2, 0.8))
    evidence = make_reset_stability_evidence(
        finite_state=True,
        grace_peak_abs_joint_velocity_rad_s=12.0,
        measured_peak_abs_joint_velocity_rad_s=0.14,
        grace_steps=30,
        measured_steps=90,
        bound_rad_s=0.5,
    )
    assert evidence["passed"] is True
    assert evidence["total_steps"] == 120
