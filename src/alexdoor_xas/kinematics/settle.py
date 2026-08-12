"""Start-pose settle postcondition (pure numpy; shared by envs and tests).

``set_ee_pose_w`` on an articulated executor can only *drive* the EE toward a
requested start pose through its kinematics — the request is not guaranteed to
be realized. This helper is the fail-closed postcondition: it measures the
realized-vs-requested residual against an explicit tolerance and produces the
provenance record (requested pose, realized pose, residual, settle ticks,
result) that episodes and eval rows carry, so pose-stratified claims always
refer to realized start states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_START_POSE_TOLERANCE_M = 0.01
"""Fail-closed position residual bound (meters). The settle loop itself
targets < 0.005 m before breaking early; twice that target (and half the
0.02 m per-tick command clamp) separates routine settle noise from a start
pose that was genuinely not reached."""


class StartPoseError(RuntimeError):
    """The requested start pose was not realized within tolerance."""

    def __init__(self, message: str, report: SettleReport):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SettleReport:
    """Realized-state evidence for one ``set_ee_pose_w`` request."""

    requested_pos_m: tuple[float, float, float]
    realized_pos_m: tuple[float, float, float]
    residual_m: float
    tolerance_m: float
    settle_ticks_used: int
    max_settle_ticks: int
    passed: bool
    orientation_checked: bool = False
    """False: orientation is not part of the request (position-mode IK
    ignores it), so no orientation residual is defined."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pos_m": list(self.requested_pos_m),
            "realized_pos_m": list(self.realized_pos_m),
            "residual_m": self.residual_m,
            "tolerance_m": self.tolerance_m,
            "settle_ticks_used": self.settle_ticks_used,
            "max_settle_ticks": self.max_settle_ticks,
            "passed": self.passed,
            "orientation_checked": self.orientation_checked,
        }


def check_settle_postcondition(
    requested_pos_m,
    realized_pos_m,
    *,
    settle_ticks_used: int,
    max_settle_ticks: int,
    tolerance_m: float = DEFAULT_START_POSE_TOLERANCE_M,
    strict: bool = True,
) -> SettleReport:
    """Measure the settle residual; fail closed when it exceeds tolerance.

    Raises :class:`StartPoseError` (``strict=True``, the default) instead of
    letting an unrealized start pose silently contaminate an episode or an
    evaluation rollout.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be finite and positive")
    requested = np.asarray(requested_pos_m, dtype=np.float64).reshape(3)
    realized = np.asarray(realized_pos_m, dtype=np.float64).reshape(3)
    finite = bool(np.isfinite(requested).all() and np.isfinite(realized).all())
    residual = float(np.linalg.norm(realized - requested)) if finite else float("nan")
    passed = finite and residual <= tolerance_m
    report = SettleReport(
        requested_pos_m=tuple(float(v) for v in requested),
        realized_pos_m=tuple(float(v) for v in realized),
        residual_m=residual,
        tolerance_m=float(tolerance_m),
        settle_ticks_used=int(settle_ticks_used),
        max_settle_ticks=int(max_settle_ticks),
        passed=passed,
    )
    if strict and not passed:
        raise StartPoseError(
            f"start pose not realized: residual {residual:.4f} m exceeds the "
            f"{tolerance_m:.4f} m tolerance after {settle_ticks_used}/"
            f"{max_settle_ticks} settle ticks (requested "
            f"{report.requested_pos_m}, realized {report.realized_pos_m})",
            report,
        )
    return report


__all__ = [
    "DEFAULT_START_POSE_TOLERANCE_M",
    "SettleReport",
    "StartPoseError",
    "check_settle_postcondition",
]
