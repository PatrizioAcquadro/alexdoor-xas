"""Start-pose settle postcondition (post-3.3 review WP9; pure helper tests)."""

from __future__ import annotations

import numpy as np
import pytest

from alexdoor_xas.data_engine import apply_start_offset
from alexdoor_xas.kinematics import (
    DEFAULT_START_POSE_TOLERANCE_M,
    StartPoseError,
    check_settle_postcondition,
)
from alexdoor_xas.policies.scripted import DoorPushVariation
from conftest import FakeDoorPushEnv


def test_realized_pose_within_tolerance_passes_and_records() -> None:
    report = check_settle_postcondition(
        (0.5, 0.2, 0.1),
        (0.503, 0.2, 0.1),
        settle_ticks_used=12,
        max_settle_ticks=90,
    )
    assert report.passed is True
    assert report.residual_m == pytest.approx(0.003)
    assert report.tolerance_m == DEFAULT_START_POSE_TOLERANCE_M
    assert report.settle_ticks_used == 12
    assert report.max_settle_ticks == 90
    assert report.orientation_checked is False  # position-only IK request
    payload = report.to_dict()
    assert payload["requested_pos_m"] == [0.5, 0.2, 0.1]
    assert payload["realized_pos_m"][0] == pytest.approx(0.503)


def test_excessive_residual_fails_closed() -> None:
    with pytest.raises(StartPoseError, match="residual 0.0300 m exceeds"):
        check_settle_postcondition(
            (0.5, 0.2, 0.1),
            (0.53, 0.2, 0.1),
            settle_ticks_used=90,
            max_settle_ticks=90,
        )


def test_non_strict_mode_records_failure_without_raising() -> None:
    report = check_settle_postcondition(
        (0.5, 0.2, 0.1),
        (0.53, 0.2, 0.1),
        settle_ticks_used=90,
        max_settle_ticks=90,
        strict=False,
    )
    assert report.passed is False
    assert report.residual_m == pytest.approx(0.03)


def test_non_finite_realized_pose_fails_closed() -> None:
    with pytest.raises(StartPoseError):
        check_settle_postcondition(
            (0.5, 0.2, 0.1),
            (float("nan"), 0.2, 0.1),
            settle_ticks_used=1,
            max_settle_ticks=90,
        )


def test_bad_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        check_settle_postcondition(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            settle_ticks_used=0,
            max_settle_ticks=90,
            tolerance_m=0.0,
        )


class _SettleReportingEnv(FakeDoorPushEnv):
    """Teleporting fake that exposes the settle-report accessor (robot-env shape)."""

    def set_proxy_pose(self, pos_w, quat_w, env_ids=None) -> None:
        super().set_proxy_pose(pos_w, quat_w, env_ids)
        requested = pos_w.detach().cpu().numpy().reshape(3)
        report = check_settle_postcondition(
            requested,
            np.asarray(self.world.ee_pos_w),
            settle_ticks_used=0,
            max_settle_ticks=90,
        )
        self._report = report.to_dict()

    def start_pose_settle_report(self):
        return self._report


def test_apply_start_offset_returns_settle_report_when_exposed() -> None:
    env = _SettleReportingEnv()
    env.reset(seed=0)
    variation = DoorPushVariation(
        start_offset_door_frame=(0.01, -0.01, 0.0), push_radius_frac=0.8, push_height_m=1.0
    )
    from alexdoor_xas.adapters import read_door_frame

    report = apply_start_offset(env, read_door_frame(env), variation)
    assert report is not None
    assert report["passed"] is True
    assert report["residual_m"] == pytest.approx(0.0)

    plain = FakeDoorPushEnv()
    plain.reset(seed=0)
    assert apply_start_offset(plain, read_door_frame(plain), variation) is None
