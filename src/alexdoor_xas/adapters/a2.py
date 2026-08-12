"""A2 adapter: world-frame end-effector delta -> executable robot command.

Adapter-v1 for ``A2_ee_delta``: validates one 6-dim per-tick EE delta (or an
``(H, 6)`` chunk of them) against the robot's execution limits, clamps it, and
returns the delta the env should execute. The env's own per-tick clamps remain
the hard back-stop — the adapter clamps *first* so the correction is decided
and logged before anything reaches the simulator.

Checks implemented here: shape/finiteness, per-tick position/rotation clamps,
workspace reachability (robots with a measured workspace model), and
warn-level joint position/velocity limit flags (envs that report joint state).
Acceleration limits, collision queries beyond the door panel, and slip
detection are documented future checks in the action-representation wiki page.
"""

from __future__ import annotations

import numpy as np

from alexdoor_xas.action.frames import panel_frame
from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .base import AdapterDecision, AdapterLog, AdapterStatus, AdapterWarning, StepContext
from .limits import RobotLimitsCfg

# Joint-limit flag bands, mirroring eval/sanity.py: unclamped dls-IK targets
# drift up to ~48 mrad past a position limit while the drive clamps at the
# limit — a known benign mode, flagged but never rejected.
JOINT_LIMIT_IGNORE_RAD = 0.01
JOINT_LIMIT_WARN_RAD = 0.1


class A2Adapter:
    """Convert predicted world-frame EE deltas into executable clamped deltas."""

    def __init__(
        self,
        limits: RobotLimitsCfg,
        log: AdapterLog | None = None,
        *,
        contact_entry_shaping: bool = False,
    ):
        self.limits = limits
        self.log = log if log is not None else AdapterLog()
        self.contact_entry_shaping = bool(contact_entry_shaping)
        self._velocity_consecutive_ticks: dict[int, int] = {}
        self._velocity_warning_counts: dict[int, int] = {}

    def process(self, delta_world, ctx: StepContext) -> tuple[np.ndarray, AdapterDecision]:
        """Adapt one 6-dim world-frame EE delta; returns (applied, decision).

        A rejected command applies zero motion (the caller still steps the env
        so tick accounting stays aligned with the policy's chunk clock).
        """
        checks: dict[str, bool] = {}
        warnings: list[str] = []
        warning_records: list[AdapterWarning] = []
        requested = np.asarray(delta_world, dtype=np.float64).reshape(-1)

        checks["shape"] = requested.shape == (EE_DELTA_DIM,)
        if not checks["shape"]:
            return self._reject(
                requested,
                checks,
                f"EE delta must have shape ({EE_DELTA_DIM},), got {requested.shape}",
            )
        checks["finite"] = bool(np.isfinite(requested).all())
        if not checks["finite"]:
            return self._reject(requested, checks, "EE delta contains non-finite values")

        applied = requested.copy()
        corrections: list[str] = []
        applied[:3] = np.clip(
            applied[:3], -self.limits.max_pos_delta_m, self.limits.max_pos_delta_m
        )
        applied[3:] = np.clip(
            applied[3:], -self.limits.max_rot_delta_rad, self.limits.max_rot_delta_rad
        )
        clamped = bool(np.any(np.abs(applied - requested) > 0.0))
        checks["within_per_tick_limits"] = not clamped
        if clamped:
            corrections.append(
                f"per-tick clamp to +/-{self.limits.max_pos_delta_m} m / "
                f"+/-{self.limits.max_rot_delta_rad} rad"
            )

        contact_shaped = self._shape_first_contact_approach(applied, ctx)
        checks["contact_approach_bounded"] = not contact_shaped
        if contact_shaped:
            corrections.append(
                "calibrated first-contact approach bound to "
                f"{self.limits.contact_approach_max_step_m} m"
            )

        checks["reachable"] = True
        if self.limits.workspace is not None:
            predicted = np.asarray(ctx.ee_pos_w, dtype=np.float64).reshape(3) + applied[:3]
            if self.limits.workspace.beyond_max_reach(predicted, self.limits.reach_margin_m):
                checks["reachable"] = False
                return self._reject(
                    requested,
                    checks,
                    f"predicted EE position {predicted.round(3).tolist()} is "
                    f"{self.limits.workspace.distance(predicted):.3f} m from the shoulder, "
                    f"beyond max reach {self.limits.workspace.max_reach_m:.3f} m",
                    warnings=warnings,
                )
            if self.limits.workspace.within_min_reach(predicted):
                warnings.append(
                    f"predicted EE position is within min reach "
                    f"{self.limits.workspace.min_reach_m:.2f} m of the shoulder "
                    "(near-singular region; IK may stall)"
                )
                warning_records.append(
                    AdapterWarning(
                        id="a2.workspace_min_reach",
                        message=warnings[-1],
                        evidence={
                            "distance_from_shoulder_m": self.limits.workspace.distance(predicted),
                            "configured_min_reach_m": self.limits.workspace.min_reach_m,
                            "tick_index": ctx.tick_index,
                            "rollout_phase": ctx.rollout_phase,
                        },
                    )
                )

        joint_warnings = self._joint_limit_flags(ctx)
        warnings.extend(warning.message for warning in joint_warnings)
        warning_records.extend(joint_warnings)

        if corrections:
            decision = AdapterDecision(
                status=AdapterStatus.CORRECTED,
                reason="; ".join(corrections),
                checks=checks,
                warnings=tuple(warnings),
                warning_records=tuple(warning_records),
                requested=requested,
                applied=applied,
            )
        else:
            decision = AdapterDecision(
                status=AdapterStatus.ACCEPTED,
                checks=checks,
                warnings=tuple(warnings),
                warning_records=tuple(warning_records),
                requested=requested,
                applied=applied,
            )
        self.log.record(decision)
        return applied, decision

    def _shape_first_contact_approach(self, applied: np.ndarray, ctx: StepContext) -> bool:
        """Bound only the unsensed inward transition through the pre-contact corridor.

        Learned A2/A3 policies have no scripted phase label. Geometry and the
        live contact flag provide the minimal phase-independent equivalent of
        Alex's calibrated scripted-controller contact approach bound: free
        space, established-contact commands, and sub-threshold sensor-dropout
        commands are unchanged. The impact trigger is twice the calibrated
        approach step: it targets the observed ~15 mm force-producing entries
        without repeatedly slowing the observed 5-10 mm post-contact motion.
        """
        limit = self.limits.contact_approach_max_step_m
        clearance = self.limits.contact_approach_start_clearance_m
        surface_x = self.limits.contact_surface_x_m
        if (
            not self.contact_entry_shaping
            or limit is None
            or clearance is None
            or surface_x is None
            or ctx.door_frame is None
            or ctx.contact_sensed is True
            or not np.isfinite(ctx.hinge_angle_rad)
        ):
            return False
        panel = panel_frame(ctx.door_frame, ctx.hinge_angle_rad)
        ee_panel = panel.point_from_world(ctx.ee_pos_w)
        delta_panel = panel.vector_from_world(applied[:3])
        translation_norm = float(np.linalg.norm(applied[:3]))
        inside_corridor = ee_panel[0] <= surface_x + clearance
        impact_risk = -delta_panel[0] > 2.0 * limit
        if not (inside_corridor and impact_risk and translation_norm > limit):
            return False
        applied[:3] *= limit / translation_norm
        return True

    def process_chunk(
        self, deltas_world, ctx: StepContext
    ) -> tuple[np.ndarray, list[AdapterDecision]]:
        """Adapt an ``(H, 6)`` action chunk against the *current* context.

        Reachability is checked against the cumulative predicted EE position,
        so a chunk that walks out of the workspace is cut at the first
        rejected step (that step and the rest apply zero motion).
        """
        deltas = np.asarray(deltas_world, dtype=np.float64)
        if deltas.ndim != 2 or deltas.shape[1] != EE_DELTA_DIM:
            raise ValueError(f"A2 chunk must have shape (H, {EE_DELTA_DIM}), got {deltas.shape}")
        applied = np.zeros_like(deltas)
        decisions: list[AdapterDecision] = []
        ee_pos = np.asarray(ctx.ee_pos_w, dtype=np.float64).reshape(3).copy()
        rejected = False
        for i, delta in enumerate(deltas):
            if rejected:
                decision = self.log.record(
                    AdapterDecision(
                        status=AdapterStatus.REJECTED,
                        reason="chunk cut: an earlier step of this chunk was rejected",
                        checks={"chunk_prefix_ok": False},
                        requested=np.asarray(delta, dtype=np.float64),
                        applied=np.zeros(EE_DELTA_DIM),
                    )
                )
                decisions.append(decision)
                continue
            step_ctx = StepContext(
                door_frame=ctx.door_frame,
                hinge_angle_rad=ctx.hinge_angle_rad,
                hinge_velocity_rad_s=ctx.hinge_velocity_rad_s,
                ee_pos_w=ee_pos.copy(),
                ee_quat_w_xyzw=ctx.ee_quat_w_xyzw,
                contact_sensed=ctx.contact_sensed,
                contact_force_n=ctx.contact_force_n,
                joint_state=ctx.joint_state if i == 0 else None,
                joint_limits=ctx.joint_limits if i == 0 else None,
                joint_names=ctx.joint_names,
                tick_index=ctx.tick_index,
                rollout_phase=ctx.rollout_phase,
            )
            step_applied, decision = self.process(delta, step_ctx)
            decisions.append(decision)
            if decision.status is AdapterStatus.REJECTED:
                rejected = True
                continue
            applied[i] = step_applied
            ee_pos += step_applied[:3]
        return applied, decisions

    def _joint_limit_flags(self, ctx: StepContext) -> list[AdapterWarning]:
        if ctx.joint_state is None or ctx.joint_limits is None:
            return []
        targets = np.asarray(ctx.joint_state.get("joint_pos_target"), dtype=np.float64)
        pos_limits = np.asarray(ctx.joint_limits.get("joint_pos_limits"), dtype=np.float64)
        warnings: list[AdapterWarning] = []
        names = ctx.joint_names or tuple(f"joint_{index}" for index in range(targets.size))
        if targets.ndim == 1 and pos_limits.shape == (targets.shape[0], 2):
            excess = np.maximum(pos_limits[:, 0] - targets, targets - pos_limits[:, 1])
            for joint in np.flatnonzero(excess > JOINT_LIMIT_IGNORE_RAD):
                joint = int(joint)
                joint_excess = float(excess[joint])
                message = (
                    f"joint target {joint} exceeds its position limit by {joint_excess:.4f} rad"
                    + (
                        ""
                        if joint_excess <= JOINT_LIMIT_WARN_RAD
                        else " (beyond the known IK-drift band)"
                    )
                )
                warnings.append(
                    AdapterWarning(
                        id="a2.joint_position_limit",
                        message=message,
                        evidence={
                            "joint_index": joint,
                            "joint_name": names[joint],
                            "tick_index": ctx.tick_index,
                            "rollout_phase": ctx.rollout_phase,
                            "exceedance_rad": joint_excess,
                            "configured_lower_limit_rad": float(pos_limits[joint, 0]),
                            "configured_upper_limit_rad": float(pos_limits[joint, 1]),
                            "target_rad": float(targets[joint]),
                        },
                    )
                )
        velocities = np.asarray(ctx.joint_state.get("joint_vel"), dtype=np.float64)
        vel_limits = np.asarray(ctx.joint_limits.get("joint_vel_limits"), dtype=np.float64)
        if velocities.shape == vel_limits.shape and velocities.size:
            over = np.abs(velocities) - vel_limits
            for index in range(velocities.size):
                if over[index] > 0.0:
                    self._velocity_consecutive_ticks[index] = (
                        self._velocity_consecutive_ticks.get(index, 0) + 1
                    )
                    self._velocity_warning_counts[index] = (
                        self._velocity_warning_counts.get(index, 0) + 1
                    )
                else:
                    self._velocity_consecutive_ticks[index] = 0
            for joint in np.flatnonzero(over > 0.0):
                joint = int(joint)
                joint_excess = float(over[joint])
                message = f"joint {joint} velocity exceeds its limit by {joint_excess:.3f} rad/s"
                warnings.append(
                    AdapterWarning(
                        id="a2.joint_velocity_limit",
                        message=message,
                        evidence={
                            "joint_index": joint,
                            "joint_name": names[joint],
                            "tick_index": ctx.tick_index,
                            "rollout_phase": ctx.rollout_phase,
                            "measured_velocity_rad_s": float(abs(velocities[joint])),
                            "configured_limit_rad_s": float(vel_limits[joint]),
                            "exceedance_rad_s": joint_excess,
                            "consecutive_ticks": self._velocity_consecutive_ticks[joint],
                            "duration_ticks": self._velocity_consecutive_ticks[joint],
                            "count": self._velocity_warning_counts[joint],
                        },
                    )
                )
        return warnings

    def _reject(
        self,
        requested: np.ndarray,
        checks: dict[str, bool],
        reason: str,
        warnings: list[str] | None = None,
    ) -> tuple[np.ndarray, AdapterDecision]:
        applied = np.zeros(EE_DELTA_DIM)
        decision = self.log.record(
            AdapterDecision(
                status=AdapterStatus.REJECTED,
                reason=reason,
                checks=checks,
                warnings=tuple(warnings or ()),
                requested=requested,
                applied=applied,
            )
        )
        return applied, decision


__all__ = ["JOINT_LIMIT_IGNORE_RAD", "JOINT_LIMIT_WARN_RAD", "A2Adapter"]
