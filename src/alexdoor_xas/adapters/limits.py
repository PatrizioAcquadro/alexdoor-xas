"""Per-robot execution limits and door-panel geometry for adapter checks.

The proxy preset is static. Alex V2 reach limits are constructed from the
validated door calibration plus a caller-supplied live shoulder center; this
module never invents or persists a workspace center.

:class:`DoorPanelGeometry` duplicates the panel constants of the scripted
controller's ``DoorPushControllerCfg`` on purpose: adapters must not import
policies, as documented in the action-representation wiki page, so a unit test
pins the two instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration

PROXY_ROBOT_TAG = "proxy_ee_sphere_v0"
ALEX_V2_ROBOT_TAG = paths.ALEX_V2_ROBOT_TAG


@dataclass(frozen=True)
class WorkspaceSphere:
    """Reachable shell around a fixed point (the fixed-base Alex shoulder)."""

    center_w: tuple[float, float, float]
    min_reach_m: float
    max_reach_m: float

    def distance(self, point_w: np.ndarray) -> float:
        center = np.asarray(self.center_w, dtype=np.float64)
        return float(np.linalg.norm(np.asarray(point_w, dtype=np.float64).reshape(3) - center))

    def beyond_max_reach(self, point_w: np.ndarray, margin_m: float = 0.0) -> bool:
        return self.distance(point_w) > self.max_reach_m + margin_m

    def within_min_reach(self, point_w: np.ndarray) -> bool:
        return self.distance(point_w) < self.min_reach_m


@dataclass(frozen=True)
class RobotLimitsCfg:
    """Execution limits one robot's adapters enforce.

    ``max_pos_delta_m`` / ``max_rot_delta_rad`` mirror the env clamps (the env
    stays the hard back-stop; the adapter clamps first and *logs* the
    correction). ``workspace`` is ``None`` for robots without a kinematic
    reach model (the velocity-driven proxy sphere).
    """

    robot: str
    max_pos_delta_m: float = 0.02
    max_rot_delta_rad: float = 0.05
    workspace: WorkspaceSphere | None = None
    reach_margin_m: float = 0.02
    """Slack on the max-reach check: commands are judged against the *measured*
    reach, which the physical EE can transiently exceed by PD tracking error."""
    contact_surface_x_m: float | None = None
    """Panel-frame X coordinate of Alex's collision-derived tool point at contact."""
    contact_approach_start_clearance_m: float | None = None
    """Calibrated align standoff where bounded first-contact approach begins."""
    contact_approach_max_step_m: float | None = None
    """Calibrated translation-norm bound for an unsensed inward contact transition."""


PROXY_LIMITS = RobotLimitsCfg(robot=PROXY_ROBOT_TAG)
"""The proxy sphere is velocity-driven and unconstrained kinematically; only
the per-tick clamps apply."""


def alex_v2_limits(
    calibration: AlexV2DoorCalibration,
    *,
    workspace_center_w,
) -> RobotLimitsCfg:
    """Build V2 limits from calibrated reach bounds and a live shoulder center."""

    center = np.asarray(workspace_center_w, dtype=np.float64).reshape(-1)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("workspace_center_w must contain exactly three finite values")
    min_reach_m, max_reach_m = calibration.reach_shell_m
    if not (
        np.isfinite(min_reach_m)
        and np.isfinite(max_reach_m)
        and 0.0 < min_reach_m < max_reach_m
    ):
        raise ValueError("calibration reach_shell_m must be finite, positive, and increasing")
    controller = calibration.controller
    contact_approach_start_clearance_m = float(controller["align_standoff_m"])
    contact_approach_max_step_m = float(controller["contact_approach_max_step_m"])
    if not (
        np.isfinite(contact_approach_start_clearance_m)
        and np.isfinite(contact_approach_max_step_m)
        and 0.0 < contact_approach_start_clearance_m
        and 0.0 < contact_approach_max_step_m <= 0.015
    ):
        raise ValueError("calibrated contact-approach limits are invalid")
    return RobotLimitsCfg(
        robot=ALEX_V2_ROBOT_TAG,
        workspace=WorkspaceSphere(
            center_w=tuple(float(value) for value in center),
            min_reach_m=float(min_reach_m),
            max_reach_m=float(max_reach_m),
        ),
        # Alex's env exposes the collision-derived tool point rather than a
        # proxy sphere center, so contact is at the physical panel thickness.
        contact_surface_x_m=DoorPanelGeometry().panel_thickness_m,
        contact_approach_start_clearance_m=contact_approach_start_clearance_m,
        contact_approach_max_step_m=contact_approach_max_step_m,
    )


def limits_for_robot(
    robot_tag: str,
    *,
    calibration: AlexV2DoorCalibration | None = None,
    workspace_center_w=None,
) -> RobotLimitsCfg:
    """Limits for a frozen robot tag, requiring live inputs for Alex V2."""

    if robot_tag == PROXY_ROBOT_TAG:
        return PROXY_LIMITS
    if robot_tag == ALEX_V2_ROBOT_TAG:
        if calibration is None or workspace_center_w is None:
            raise ValueError(
                "Alex V2 limits require validated calibration and workspace_center_w"
            )
        return alex_v2_limits(calibration, workspace_center_w=workspace_center_w)
    raise KeyError(
        f"no adapter limits for robot {robot_tag!r} "
        f"(known: {[ALEX_V2_ROBOT_TAG, PROXY_ROBOT_TAG]})"
    )


@dataclass(frozen=True)
class DoorPanelGeometry:
    """Measured door-panel geometry in the panel frame.

    The panel occupies ``x in [0, thickness]``, ``y in [0, width]``,
    ``z in [-height/2, height/2]`` (the hinge origin sits at panel mid-height).
    The handle protrudes from the +X push face inside the handle band.
    """

    panel_width_m: float = 0.83
    panel_height_m: float = 2.0
    panel_thickness_m: float = 0.036
    ee_radius_m: float = 0.05
    handle_band_y_m: tuple[float, float] = (0.63, 0.80)
    handle_band_z_m: tuple[float, float] = (0.0, 0.09)
    contact_eps_m: float = 0.002

    def surface_x_m(self, clearance_m: float) -> float:
        """Panel-frame x of the EE center at ``clearance_m`` off the +X face."""
        return self.panel_thickness_m + self.ee_radius_m + clearance_m

    def on_panel(self, point_panel: np.ndarray, tol_m: float = 0.0) -> bool:
        """Whether a panel-frame contact point lies on the panel face (±tol)."""
        point = np.asarray(point_panel, dtype=np.float64).reshape(3)
        half_height = self.panel_height_m / 2.0
        return bool(
            -tol_m <= point[1] <= self.panel_width_m + tol_m
            and -half_height - tol_m <= point[2] <= half_height + tol_m
        )

    def clamp_to_panel(self, point_panel: np.ndarray) -> np.ndarray:
        """Nudge a panel-frame point onto the panel face (y/z bounds only)."""
        point = np.asarray(point_panel, dtype=np.float64).reshape(3).copy()
        half_height = self.panel_height_m / 2.0
        point[1] = min(max(point[1], 0.0), self.panel_width_m)
        point[2] = min(max(point[2], -half_height), half_height)
        return point

    def in_handle_band(self, point_panel: np.ndarray) -> bool:
        point = np.asarray(point_panel, dtype=np.float64).reshape(3)
        return bool(
            self.handle_band_y_m[0] <= point[1] <= self.handle_band_y_m[1]
            and self.handle_band_z_m[0] <= point[2] <= self.handle_band_z_m[1]
        )

    def geometric_contact(self, ee_panel: np.ndarray) -> bool:
        """EE-surface-on-panel-face inference (the scripted controller's rule)."""
        point = np.asarray(ee_panel, dtype=np.float64).reshape(3)
        on_face = point[0] <= self.surface_x_m(self.contact_eps_m)
        within_panel = (
            -self.ee_radius_m <= point[1] <= self.panel_width_m + self.ee_radius_m
            and point[0] >= -self.ee_radius_m
        )
        return bool(on_face and within_panel)


MAX_HINGE_ANGLE_RAD = math.pi / 2.0
"""Physical hinge travel: the door stops against the wall at 90 degrees."""


__all__ = [
    "ALEX_V2_ROBOT_TAG",
    "MAX_HINGE_ANGLE_RAD",
    "PROXY_LIMITS",
    "PROXY_ROBOT_TAG",
    "DoorPanelGeometry",
    "RobotLimitsCfg",
    "WorkspaceSphere",
    "alex_v2_limits",
    "limits_for_robot",
]
