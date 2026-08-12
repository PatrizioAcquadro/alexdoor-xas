"""Runtime calibration contract tests."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alexdoor_xas import paths
from alexdoor_xas.calibration.alex_v2_door import (
    CalibrationError,
    load_alex_v2_door_calibration,
)


def _payload() -> dict:
    return json.loads(paths.ALEX_V2_CALIBRATION.read_text(encoding="utf-8"))


def _write(path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_active_calibration() -> None:
    calibration = load_alex_v2_door_calibration(paths.ALEX_V2_CALIBRATION)

    assert calibration.base_pose["position_m"] == pytest.approx((-0.45, -0.38, 0.93))
    assert set(calibration.ready_joint_pos) == {
        "RIGHT_SHOULDER_Y",
        "RIGHT_SHOULDER_X",
        "RIGHT_SHOULDER_Z",
        "RIGHT_ELBOW_Y",
        "RIGHT_WRIST_Z",
        "RIGHT_WRIST_X",
    }
    assert calibration.tool_frame["parent_link"] == "RIGHT_GRIPPER_Z_LINK"
    assert calibration.reach_shell_m == pytest.approx((0.2, 0.8))


@pytest.mark.parametrize(
    ("field_path", "value", "match"),
    [
        (("schema_version",), "wrong", "schema is unsupported"),
        (("task",), "wrong", "task must be"),
        (("tool_frame", "orientation_xyzw"), [0.0, 0.0, 0.0, 2.0], "unit length"),
        (("ready_joint_pos", "RIGHT_ELBOW_Y"), 1.0, "outside"),
        (("reach_shell_m",), [0.8, 0.2], "positive and increasing"),
        (("controller", "contact_force_threshold_n"), float("nan"), "finite"),
        (("controller", "align_standoff_m"), 0.13, "no greater than"),
        (("controller", "contact_approach_max_step_m"), 0.02, "must be in"),
        (("randomization_bounds", "start_offset_low"), [0.01, -0.02, -0.02], "include zero"),
        (("randomization_bounds", "push_radius_frac_range"), [0.4, 0.5], "excludes"),
    ],
)
def test_rejects_invalid_semantics(tmp_path, field_path, value, match) -> None:
    payload = deepcopy(_payload())
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    source = tmp_path / "calibration.json"
    _write(source, payload)

    with pytest.raises(CalibrationError, match=match):
        load_alex_v2_door_calibration(source)


def test_rejects_unknown_fields_and_unreadable_payloads(tmp_path) -> None:
    payload = _payload()
    payload["status"] = "validated"
    source = tmp_path / "calibration.json"
    _write(source, payload)
    with pytest.raises(CalibrationError, match="unexpected status"):
        load_alex_v2_door_calibration(source)

    _write(source, [])
    with pytest.raises(CalibrationError, match="JSON object"):
        load_alex_v2_door_calibration(source)

    with pytest.raises(CalibrationError, match="cannot load"):
        load_alex_v2_door_calibration(tmp_path / "missing.json")
