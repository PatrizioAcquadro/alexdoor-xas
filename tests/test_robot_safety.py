"""Start-pose settle safety contracts."""

from __future__ import annotations

import pytest

from alexdoor_xas.data_engine import apply_start_offset
from alexdoor_xas.kinematics.settle import StartPoseError, validate_start_pose_settle
from alexdoor_xas.policies.scripted.door_push import DoorPushVariation
from conftest import FakeDoorPushEnv


def test_realized_pose_within_tolerance_passes_and_records() -> None:
    report = validate_start_pose_settle(
        (0.5, 0.2, 0.1),
        (0.503, 0.2, 0.1),
        settle_ticks_used=12,
        max_settle_ticks=90,
        tolerance_m=0.01,
    )
    assert report.to_dict() == {
        "requested_pos_m": [0.5, 0.2, 0.1],
        "realized_pos_m": [0.503, 0.2, 0.1],
        "residual_m": pytest.approx(0.003),
        "tolerance_m": 0.01,
        "settle_ticks_used": 12,
        "max_settle_ticks": 90,
        "passed": True,
        "orientation_checked": False,
    }


def test_excessive_residual_fails_closed() -> None:
    with pytest.raises(StartPoseError, match="residual 0.0300 m exceeds") as raised:
        validate_start_pose_settle(
            (0.5, 0.2, 0.1),
            (0.53, 0.2, 0.1),
            settle_ticks_used=90,
            max_settle_ticks=90,
            tolerance_m=0.01,
        )
    assert raised.value.report.to_dict() == {
        "requested_pos_m": [0.5, 0.2, 0.1],
        "realized_pos_m": [0.53, 0.2, 0.1],
        "residual_m": pytest.approx(0.03),
        "tolerance_m": 0.01,
        "settle_ticks_used": 90,
        "max_settle_ticks": 90,
        "passed": False,
        "orientation_checked": False,
    }

def test_non_finite_realized_pose_fails_closed() -> None:
    with pytest.raises(StartPoseError):
        validate_start_pose_settle(
            (0.5, 0.2, 0.1),
            (float("nan"), 0.2, 0.1),
            settle_ticks_used=1,
            max_settle_ticks=90,
            tolerance_m=0.01,
        )


def test_bad_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        validate_start_pose_settle(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            settle_ticks_used=0,
            max_settle_ticks=90,
            tolerance_m=0.0,
        )


def test_apply_start_offset_returns_required_settle_report() -> None:
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    variation = DoorPushVariation(
        start_offset_door_frame=(0.01, -0.01, 0.0), push_radius_frac=0.8, push_height_m=1.0
    )
    from alexdoor_xas.adapters.rollout import read_door_frame

    report = apply_start_offset(env, read_door_frame(env), variation)
    assert report is not None
    assert report["passed"] is True
    assert report["residual_m"] == pytest.approx(0.0)
