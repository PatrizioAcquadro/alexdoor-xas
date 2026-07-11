"""Pure fail-closed tests for Alex V2 door calibration."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.calibration.alex_v2_door import (
    CALIBRATION_SCHEMA,
    REQUIRED_GATES,
    CalibrationError,
    calibration_fingerprint,
    load_candidate_alex_v2_door_calibration,
    load_validated_alex_v2_door_calibration,
)


def _payload(asset: RobotAssetRef, runtime: dict[str, str]) -> dict:
    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "status": "validated",
        "task": "door_push_alex_v2",
        "robot_asset": asset.to_dict(),
        "runtime_versions": runtime,
        "tool_frame": {
            "parent_link": "RIGHT_GRIPPER_Z_LINK",
            "translation_m": [0.11, 0.0, -0.06],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "contact_normal_link": [1.0, 0.0, 0.0],
            "support_shape": "right_thumb_collision",
            "support_distance_m": 0.11,
            "collision_union_sha256": "c" * 64,
        },
        "base_pose": {
            "position_m": [-0.45, -0.38, 0.93],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        "ready_joint_pos": {
            "RIGHT_SHOULDER_Y": 0.2,
            "RIGHT_SHOULDER_X": -0.2,
            "RIGHT_SHOULDER_Z": 0.0,
            "RIGHT_ELBOW_Y": -0.7,
            "RIGHT_WRIST_Z": 0.0,
            "RIGHT_WRIST_X": 0.0,
        },
        # Synthetic test bounds only. Production values require a measured V2 arc.
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
        "gates": {name: True for name in REQUIRED_GATES},
    }
    payload["fingerprint"] = calibration_fingerprint(payload)
    return payload


def _write(path, payload: dict) -> None:
    payload["fingerprint"] = calibration_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _assert_rejected(path, payload, asset, runtime, match: str) -> None:
    _write(path, payload)
    with pytest.raises(CalibrationError, match=match):
        load_validated_alex_v2_door_calibration(
            path, runtime_asset=asset, runtime_versions=runtime
        )


def test_validated_calibration_requires_exact_asset_runtime_gates_and_hash(
    tmp_path,
) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_payload(asset, runtime)), encoding="utf-8")

    loaded = load_validated_alex_v2_door_calibration(
        path, runtime_asset=asset, runtime_versions=runtime
    )
    assert loaded.robot_asset == asset
    assert loaded.reach_shell_m == pytest.approx((0.2, 0.8))

    candidate = _payload(asset, runtime)
    candidate["status"] = "candidate"
    candidate["fingerprint"] = calibration_fingerprint(candidate)
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(CalibrationError, match="not validated"):
        load_validated_alex_v2_door_calibration(
            path, runtime_asset=asset, runtime_versions=runtime
        )


def test_candidate_loader_allows_incomplete_gates_only_for_candidate_status(
    tmp_path,
) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "candidate.json"
    candidate = _payload(asset, runtime)
    candidate["status"] = "candidate"
    candidate["gates"]["contact_behavior"] = False
    _write(path, candidate)

    loaded = load_candidate_alex_v2_door_calibration(
        path,
        runtime_asset=asset,
        runtime_versions=runtime,
    )
    assert loaded.status == "candidate"
    assert loaded.payload["gates"]["contact_behavior"] is False

    with pytest.raises(CalibrationError, match="not validated"):
        load_validated_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions=runtime,
        )

    validated = _payload(asset, runtime)
    _write(path, validated)
    with pytest.raises(CalibrationError, match="must have status='candidate'"):
        load_candidate_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions=runtime,
        )


def test_candidate_loader_still_requires_complete_structure_and_boolean_gates(
    tmp_path,
) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "candidate.json"

    missing_gate = _payload(asset, runtime)
    missing_gate["status"] = "candidate"
    missing_gate["gates"].pop("contact_behavior")
    _write(path, missing_gate)
    with pytest.raises(CalibrationError, match="gates fields must match exactly"):
        load_candidate_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions=runtime,
        )

    non_boolean_gate = _payload(asset, runtime)
    non_boolean_gate["status"] = "candidate"
    non_boolean_gate["gates"]["contact_behavior"] = 1
    _write(path, non_boolean_gate)
    with pytest.raises(CalibrationError, match="must be a boolean"):
        load_candidate_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions=runtime,
        )

    invalid_tool = _payload(asset, runtime)
    invalid_tool["status"] = "candidate"
    invalid_tool["tool_frame"]["orientation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
    _write(path, invalid_tool)
    with pytest.raises(CalibrationError, match="orientation_xyzw must be unit length"):
        load_candidate_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions=runtime,
        )


def test_calibration_rejects_missing_gate_stale_runtime_and_tampering(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"
    payload = _payload(asset, runtime)
    payload["gates"]["contact_behavior"] = False
    payload["fingerprint"] = calibration_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="contact_behavior"):
        load_validated_alex_v2_door_calibration(
            path, runtime_asset=asset, runtime_versions=runtime
        )

    payload = _payload(asset, runtime)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="runtime versions"):
        load_validated_alex_v2_door_calibration(
            path,
            runtime_asset=asset,
            runtime_versions={**runtime, "isaac_lab": "3.1.0"},
        )

    payload["controller"]["push_height_m"] = 0.16
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="fingerprint mismatch"):
        load_validated_alex_v2_door_calibration(
            path, runtime_asset=asset, runtime_versions=runtime
        )


def test_missing_calibration_fails_closed(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    with pytest.raises(CalibrationError, match="unavailable"):
        load_validated_alex_v2_door_calibration(
            tmp_path / "missing.json",
            runtime_asset=asset,
            runtime_versions={"isaac_sim": "6.0.1", "isaac_lab": "3.0.0"},
        )


def test_calibration_rejects_wrong_task_and_invalid_frame_geometry(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"

    cases = (
        (("task",), "invalid_task", "task must be exactly"),
        (
            ("tool_frame", "orientation_xyzw"),
            [0.0, 0.0, 0.0, 2.0],
            "orientation_xyzw must be unit length",
        ),
        (
            ("tool_frame", "contact_normal_link"),
            [2.0, 0.0, 0.0],
            "contact_normal_link must be unit length",
        ),
        (
            ("tool_frame", "collision_union_sha256"),
            "C" * 64,
            "collision union fingerprint is invalid",
        ),
        (
            ("tool_frame", "support_distance_m"),
            0.0,
            "support_distance_m must be positive",
        ),
        (
            ("base_pose", "orientation_xyzw"),
            [0.0, 0.0, 0.0, 0.0],
            "orientation_xyzw must be unit length",
        ),
    )
    for keys, value, match in cases:
        payload = deepcopy(_payload(asset, runtime))
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        _assert_rejected(path, payload, asset, runtime, match)


def test_calibration_requires_exact_six_joint_ready_pose_with_official_limits(
    tmp_path,
) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"

    missing = _payload(asset, runtime)
    missing["ready_joint_pos"].pop("RIGHT_WRIST_X")
    _assert_rejected(
        path, missing, asset, runtime, "ready_joint_pos fields must match exactly"
    )

    extra = _payload(asset, runtime)
    extra["ready_joint_pos"]["RIGHT_GRIPPER_Z"] = 0.0
    _assert_rejected(path, extra, asset, runtime, "unexpected RIGHT_GRIPPER_Z")

    outside = _payload(asset, runtime)
    outside["ready_joint_pos"]["RIGHT_ELBOW_Y"] = 0.2
    _assert_rejected(path, outside, asset, runtime, "outside official V2 limits")


@pytest.mark.parametrize(
    ("value", "match"),
    (
        ([0.2], "must contain 2 numbers"),
        ([0.2, 0.8, 0.9], "must contain 2 numbers"),
        ([0.0, 0.8], "minimum must be positive"),
        ([-0.1, 0.8], "minimum must be positive"),
        ([0.8, 0.8], "minimum must be less than maximum"),
        ([0.9, 0.8], "minimum must be less than maximum"),
        ([0.2, float("inf")], "must be finite"),
        ([True, 0.8], "must be a number"),
    ),
)
def test_calibration_rejects_invalid_reach_shell(tmp_path, value, match: str) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"
    payload = _payload(asset, runtime)
    payload["reach_shell_m"] = value

    _assert_rejected(path, payload, asset, runtime, match)


def test_calibration_requires_reach_shell_field(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"
    payload = _payload(asset, runtime)
    payload.pop("reach_shell_m")

    _assert_rejected(path, payload, asset, runtime, "must contain 2 numbers")


def test_calibration_rejects_invalid_controller_contract(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"

    cases = (
        ("extra", 1.0, "controller fields must match exactly"),
        ("push_radius_frac", 0.0, "push_radius_frac must be in"),
        ("push_height_m", float("nan"), "push_height_m must be finite"),
        ("align_standoff_m", 0.13, "align_standoff_m must be in"),
        (
            "contact_approach_max_step_m",
            0.0,
            "contact_approach_max_step_m must be in",
        ),
        ("contact_clearance_m", -0.051, "contact_clearance_m must be in"),
        ("release_standoff_m", 0.01, "release_standoff_m must be in"),
        ("contact_force_threshold_n", 0.0, "contact_force_threshold_n must be in"),
    )
    for field, value, match in cases:
        payload = deepcopy(_payload(asset, runtime))
        payload["controller"][field] = value
        _assert_rejected(path, payload, asset, runtime, match)


def test_calibration_rejects_invalid_randomization_contract(tmp_path) -> None:
    asset = RobotAssetRef("alex-v2", "a" * 64, "b" * 64)
    runtime = {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"}
    path = tmp_path / "calibration.json"

    extra = _payload(asset, runtime)
    extra["randomization_bounds"]["unexpected"] = [0.0, 1.0]
    _assert_rejected(
        path, extra, asset, runtime, "randomization_bounds fields must match exactly"
    )

    reversed_offset = _payload(asset, runtime)
    reversed_offset["randomization_bounds"]["start_offset_low"][0] = 0.03
    _assert_rejected(path, reversed_offset, asset, runtime, "low greater than high")

    excludes_zero = _payload(asset, runtime)
    excludes_zero["randomization_bounds"]["start_offset_low"][1] = 0.001
    _assert_rejected(path, excludes_zero, asset, runtime, "excludes nominal zero")

    reversed_range = _payload(asset, runtime)
    reversed_range["randomization_bounds"]["push_height_m_range"] = [0.2, 0.1]
    _assert_rejected(path, reversed_range, asset, runtime, "low greater than high")

    excludes_nominal = _payload(asset, runtime)
    excludes_nominal["randomization_bounds"]["push_radius_frac_range"] = [0.36, 0.40]
    _assert_rejected(
        path, excludes_nominal, asset, runtime, "excludes nominal controller"
    )
