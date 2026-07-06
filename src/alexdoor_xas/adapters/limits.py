"""Per-robot execution limits and door-panel geometry for adapter checks.

The limit presets are keyed by the frozen robot tags (docs/action_spaces.md).
The Alex workspace numbers are the Phase 2.5 *measured* constants (probe
experiments, docs/phase2_5_alex_report.md) — do not retune them here without
rerunning ``scripts/verify_alex_ik_probe.py``.

:class:`DoorPanelGeometry` duplicates the panel constants of the scripted
controller's ``DoorPushControllerCfg`` on purpose: adapters must not import
policies (docs/architecture.md), so a unit test pins the two instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PROXY_ROBOT_TAG = "proxy_ee_sphere_v0"
ALEX_ROBOT_TAG = "alex_v1_fullbody_fixedbase_v0"


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


PROXY_LIMITS = RobotLimitsCfg(robot=PROXY_ROBOT_TAG)
"""The proxy sphere is velocity-driven and unconstrained kinematically; only
the per-tick clamps apply."""

ALEX_LIMITS = RobotLimitsCfg(
    robot=ALEX_ROBOT_TAG,
    workspace=WorkspaceSphere(
        # Measured in Phase 2.5: right SHOULDER_Z link at world (-0.43, -0.10,
        # 1.39); usable reach 0.584 m. A waypoint 0.218 m from the shoulder
        # timed out (arm folded near-singular) while the working push arc
        # starts at 0.25 m — 0.24 m splits the measured pass/fail boundary.
        center_w=(-0.43, -0.10, 1.39),
        min_reach_m=0.24,
        max_reach_m=0.584,
    ),
)

_LIMITS_BY_ROBOT = {cfg.robot: cfg for cfg in (PROXY_LIMITS, ALEX_LIMITS)}


def limits_for_robot(robot_tag: str) -> RobotLimitsCfg:
    """Limit preset for a frozen robot tag; raises on unknown tags."""
    try:
        return _LIMITS_BY_ROBOT[robot_tag]
    except KeyError:
        raise KeyError(
            f"no adapter limits for robot {robot_tag!r} (known: {sorted(_LIMITS_BY_ROBOT)})"
        ) from None


@dataclass(frozen=True)
class DoorPanelGeometry:
    """Measured door-panel geometry in the panel frame (docs/phase2_report.md).

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
    "ALEX_LIMITS",
    "ALEX_ROBOT_TAG",
    "MAX_HINGE_ANGLE_RAD",
    "PROXY_LIMITS",
    "PROXY_ROBOT_TAG",
    "DoorPanelGeometry",
    "RobotLimitsCfg",
    "WorkspaceSphere",
    "limits_for_robot",
]
