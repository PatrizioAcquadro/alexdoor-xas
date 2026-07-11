"""Pure tests for the Alex V2 calibration author/validation gate helpers."""

from __future__ import annotations

import json

import pytest

from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.calibration.alex_v2_door import (
    REQUIRED_GATES,
    load_candidate_alex_v2_door_calibration,
    load_validated_alex_v2_door_calibration,
)
from alexdoor_xas.calibration.alex_v2_door_authoring import (
    CalibrationAuthoringError,
    compose_candidate_payload,
    distance_envelope,
    envelope_within_shell,
    make_reset_stability_evidence,
    make_validated_payload,
    write_validated_calibration,
)


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


@pytest.fixture
def identity() -> tuple[RobotAssetRef, dict[str, str]]:
    return (
        RobotAssetRef("alex-v2-test", "a" * 64, "b" * 64),
        {"isaac_sim": "6.0.1-rc.7", "isaac_lab": "3.0.0"},
    )


def _candidate(identity: tuple[RobotAssetRef, dict[str, str]]) -> dict:
    asset, runtime = identity
    return compose_candidate_payload(
        _manifest(),
        robot_asset=asset,
        runtime_versions=runtime,
    )


def test_composed_candidate_passes_candidate_loader(tmp_path, identity) -> None:
    asset, runtime = identity
    payload = _candidate(identity)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_candidate_alex_v2_door_calibration(
        path,
        runtime_asset=asset,
        runtime_versions=runtime,
    )

    assert loaded.status == "candidate"
    assert loaded.payload["tool_frame"]["parent_link"] == "RIGHT_GRIPPER_Z_LINK"
    assert loaded.payload["tool_frame"]["translation_m"] == pytest.approx(
        [0.12, 0.0, -0.06]
    )
    assert loaded.payload["ready_joint_pos"]["RIGHT_SHOULDER_Y"] == pytest.approx(0.3)
    assert loaded.payload["controller"] == {
        "push_radius_frac": 0.35,
        "push_height_m": 0.15,
        "approach_standoff_m": 0.12,
        "align_standoff_m": 0.10,
        "pre_contact_clearance_m": 0.01,
        "contact_clearance_m": -0.002,
        "contact_approach_max_step_m": 0.005,
        "release_standoff_m": 0.3,
        "contact_force_threshold_n": 1.5,
    }
    assert loaded.payload["randomization_bounds"] == {
        "start_offset_low": [-0.02, -0.02, -0.02],
        "start_offset_high": [0.02, 0.02, 0.02],
        "push_radius_frac_range": [0.33, 0.37],
        "push_height_m_range": [0.12, 0.17],
    }
    assert loaded.payload["gates"] == {name: False for name in REQUIRED_GATES}


def test_all_true_gates_produce_loadable_validated_payload(tmp_path, identity) -> None:
    asset, runtime = identity
    gates = {name: True for name in REQUIRED_GATES}
    payload = make_validated_payload(_candidate(identity), gates)
    path = tmp_path / "validated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_validated_alex_v2_door_calibration(
        path,
        runtime_asset=asset,
        runtime_versions=runtime,
    )

    assert loaded.status == "validated"
    assert loaded.payload["gates"] == gates


@pytest.mark.parametrize("failed_gate", REQUIRED_GATES)
def test_production_write_refuses_each_failed_gate(tmp_path, identity, failed_gate) -> None:
    destination = tmp_path / "configs" / "alex_v2_door_calibration.v0.json"
    gates = {name: True for name in REQUIRED_GATES}
    gates[failed_gate] = False

    with pytest.raises(CalibrationAuthoringError, match=failed_gate):
        write_validated_calibration(destination, _candidate(identity), gates)

    assert not destination.exists()


def test_production_writer_authors_loadable_payload_atomically(tmp_path, identity) -> None:
    asset, runtime = identity
    destination = tmp_path / "configs" / "alex_v2_door_calibration.v0.json"
    gates = {name: True for name in REQUIRED_GATES}

    written = write_validated_calibration(destination, _candidate(identity), gates)
    loaded = load_validated_alex_v2_door_calibration(
        destination,
        runtime_asset=asset,
        runtime_versions=runtime,
    )

    assert written == loaded.payload
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_distance_envelope_and_shell_containment() -> None:
    assert distance_envelope([0.4, 0.2, 0.7]) == pytest.approx((0.2, 0.7))
    assert envelope_within_shell([0.2, 0.4, 0.8], (0.2, 0.8))
    assert not envelope_within_shell([0.19, 0.4], (0.2, 0.8))
    assert not envelope_within_shell([0.4, 0.81], (0.2, 0.8))
    assert not envelope_within_shell([], (0.2, 0.8))
    assert not envelope_within_shell([0.4, float("nan")], (0.2, 0.8))


def test_reset_stability_uses_only_the_post_grace_velocity_window() -> None:
    evidence = make_reset_stability_evidence(
        finite_state=True,
        grace_peak_abs_joint_velocity_rad_s=12.048294,
        measured_peak_abs_joint_velocity_rad_s=0.144745,
        grace_steps=30,
        measured_steps=90,
        bound_rad_s=0.5,
    )

    assert evidence == {
        "finite_state": True,
        "grace_peak_abs_joint_velocity_rad_s": pytest.approx(12.048294),
        "peak_abs_joint_velocity_rad_s": pytest.approx(0.144745),
        "bound_rad_s": pytest.approx(0.5),
        "grace_steps": 30,
        "measured_steps": 90,
        "total_steps": 120,
        "passed": True,
    }

    evidence = make_reset_stability_evidence(
        finite_state=True,
        grace_peak_abs_joint_velocity_rad_s=0.1,
        measured_peak_abs_joint_velocity_rad_s=0.5,
        grace_steps=30,
        measured_steps=90,
        bound_rad_s=0.5,
    )
    assert evidence["passed"] is False

    no_grace = make_reset_stability_evidence(
        finite_state=True,
        grace_peak_abs_joint_velocity_rad_s=0.0,
        measured_peak_abs_joint_velocity_rad_s=0.49,
        grace_steps=0,
        measured_steps=120,
        bound_rad_s=0.5,
    )
    assert no_grace["passed"] is True
    assert no_grace["total_steps"] == 120
