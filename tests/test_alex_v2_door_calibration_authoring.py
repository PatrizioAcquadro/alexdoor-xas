"""Pure tests for live-gated Alex V2 calibration authoring."""

from __future__ import annotations

import json

import pytest

from alexdoor_xas.calibration.alex_v2_door import load_alex_v2_door_calibration
from alexdoor_xas.calibration.alex_v2_door_authoring import (
    REQUIRED_GATES,
    CalibrationAuthoringError,
    compose_calibration_payload,
    distance_envelope,
    envelope_within_shell,
    make_reset_stability_evidence,
    write_calibration,
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
