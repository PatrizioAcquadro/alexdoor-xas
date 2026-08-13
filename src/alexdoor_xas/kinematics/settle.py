"""Fail-closed validation for realized start poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class StartPoseError(RuntimeError):
    """The requested start pose was not realized within tolerance."""

    def __init__(self, message: str, report: SettleReport):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SettleReport:
    """Evidence for one realized start-pose request."""

    requested_pos_m: tuple[float, float, float]
    realized_pos_m: tuple[float, float, float]
    residual_m: float
    tolerance_m: float
    settle_ticks_used: int
    max_settle_ticks: int
    passed: bool
    orientation_checked: bool = False

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


def validate_start_pose_settle(
    requested_pos_m,
    realized_pos_m,
    *,
    settle_ticks_used: int,
    max_settle_ticks: int,
    tolerance_m: float,
) -> SettleReport:
    """Return settle evidence or raise when the realized position is invalid."""
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
    if not passed:
        raise StartPoseError(
            f"start pose not realized: residual {residual:.4f} m exceeds the "
            f"{tolerance_m:.4f} m tolerance after {settle_ticks_used}/"
            f"{max_settle_ticks} settle ticks (requested "
            f"{report.requested_pos_m}, realized {report.realized_pos_m})",
            report,
        )
    return report
