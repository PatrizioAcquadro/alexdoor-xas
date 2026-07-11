"""Rollout sanity checks for force-sensing (Alex) episodes (pure numpy).

Catches silently bad recorded data before it reaches Phase 3 dataset loaders:
non-finite joint state, IK targets outside the robot's joint limits, runaway
joint velocities, missing force sensing, and suspicious force spikes. The
checks read only the episode schema (no Isaac imports): joint limits come from
the ``joint_pos_limits`` / ``joint_vel_limits`` extras the data engine records
when the env exposes ``robot_joint_limits()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alexdoor_xas.recording import EpisodeBuffer

CONTACT_SOURCE_FORCE = "force_sensor+geometric"

MAX_ARM_JOINT_VEL_RAD_S = 4.0
"""Task-level arm velocity warning cap: the scripted push commands <= 0.02 m
per 1/60 s tick, which the 6 IK arm joints track well below this rate; getting
near the vendored sim limits (10-25 rad/s) would mean instability, not pushing."""

FORCE_WARN_N = 200.0
"""Contact-force warning threshold: measured gate pushes peak at tens of newtons
against the 25 kg damped door; hundreds indicate a jammed or unstable contact."""

FORCE_DATASET_LIMIT_N = FORCE_WARN_N
"""Hard Stage 3/Stage 4 admission limit, intentionally equal to the warning threshold.

General diagnostic callers still receive a warning by default. Calibration and dataset
writers opt into a hard error at this same value; they may not raise it to admit a run.
"""

SETTLE_TICKS = 30
"""Velocity checks skip this initial window (0.5 s at control_dt = 1/60): the
position-held legs spike for the first ~16 ticks after reset while the PD
catches the standing pose (measured on the gate: knee/ankle peaks 9.5-11.2
rad/s at ticks 1-16, then <= 2.1 rad/s for the rest of the rollout). Episodes
whose recording starts after the IK settle loop show no transient at all."""

LIMIT_MARGIN_RAD = 0.01
"""Ignored joint-target excursion past the position limits: the differential IK
does not clamp its targets to joint limits, and sub-centirad overshoot is
routine dls jitter (measured 3.2 mrad on the headless gate)."""

LIMIT_ERROR_RAD = 0.1
"""Target overshoot beyond which the episode is an error, not a warning. The
IK can wind targets past a limit while the drive is clamped at it (measured up
to 48 mrad on a camera-enabled randomized episode — harmless, the arm tracks
the limit); an excursion past ~0.1 rad means sustained windup or bad data."""

_PROPRIO_KEYS = ("joint_pos", "joint_vel", "joint_pos_target")


@dataclass
class SanityResult:
    """Outcome of :func:`check_alex_episode`: hard failures + soft warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def contact_force_diagnostics(
    episode: EpisodeBuffer,
    *,
    force_limit_n: float = FORCE_DATASET_LIMIT_N,
    window_radius: int = 3,
) -> dict:
    """Return bounded, JSON-able evidence for one episode's force admission.

    Force is sampled before the current step's action is applied, so
    ``causal_action_phase`` and ``causal_action_world`` intentionally refer to
    tick ``t-1``. The current tick carries the resulting pose, door state,
    joint state, contact state, and filtered force.
    """

    limit = float(force_limit_n)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("force_limit_n must be finite and positive")
    if isinstance(window_radius, bool) or not isinstance(window_radius, int) or window_radius < 0:
        raise ValueError("window_radius must be a non-negative integer")
    forces = np.asarray(
        [float(step.contact.get("force_n", 0.0)) for step in episode.steps],
        dtype=np.float64,
    )
    if not forces.size:
        return {
            "all_forces_finite": True,
            "min_force_n": None,
            "min_force_tick": None,
            "max_force_n": None,
            "max_force_tick": None,
            "force_limit_n": limit,
            "upper_force_gate_passed": False,
            "non_negative_force_gate_passed": False,
            "force_admission_passed": False,
            "non_finite_force_ticks": [],
            "negative_force_ticks": [],
            "ticks_over_limit": [],
            "one_tick_over_limit": False,
            "sustained_over_limit": False,
            "peak": None,
            "window": [],
            "ik_clamp_telemetry": episode.extras.get("ik_clamp_telemetry"),
        }

    finite = np.isfinite(forces)
    has_finite = bool(finite.any())
    safe_for_argmax = np.where(finite, forces, -np.inf)
    safe_for_argmin = np.where(finite, forces, np.inf)
    peak_tick = int(np.argmax(safe_for_argmax)) if has_finite else None
    minimum_tick = int(np.argmin(safe_for_argmin)) if has_finite else None
    non_finite_ticks = np.flatnonzero(~finite).astype(int).tolist()
    negative_ticks = np.flatnonzero(finite & (forces < 0.0)).astype(int).tolist()
    over_limit = np.flatnonzero(finite & (forces > limit)).astype(int).tolist()
    sustained = any(
        current == previous + 1
        for previous, current in zip(over_limit, over_limit[1:], strict=False)
    )
    arm_ids = [int(value) for value in episode.extras.get("arm_joint_ids", [])]
    joint_names = [str(value) for value in episode.extras.get("joint_names", [])]

    def joint_name(index: int) -> str:
        return joint_names[index] if index < len(joint_names) else f"joint[{index}]"

    def sample(tick: int) -> dict:
        step = episode.steps[tick]
        causal_tick = tick - 1
        causal_step = episode.steps[causal_tick] if causal_tick >= 0 else None
        proprio = step.proprio

        def arm_values(key: str) -> dict[str, float] | None:
            if key not in proprio or not arm_ids:
                return None
            values = np.asarray(proprio[key], dtype=np.float64).reshape(-1)
            return {joint_name(index): float(values[index]) for index in arm_ids}

        return {
            "tick": tick,
            "t_s": float(step.t),
            "force_n": float(forces[tick]),
            "contact_sensed": step.contact.get("sensed"),
            "contact_inferred": step.contact.get("inferred"),
            "recorded_phase": str(step.safety.get("controller_phase", "")),
            "causal_action_tick": causal_tick if causal_step is not None else None,
            "causal_action_phase": (
                str(causal_step.safety.get("controller_phase", ""))
                if causal_step is not None
                else None
            ),
            "causal_action_world": (
                np.asarray(causal_step.action, dtype=np.float64).tolist()
                if causal_step is not None
                else None
            ),
            "tool_position_world_m": (
                np.asarray(proprio["ee_pos_w"], dtype=np.float64).tolist()
                if "ee_pos_w" in proprio
                else None
            ),
            "tool_orientation_world_xyzw": (
                np.asarray(proprio["ee_quat_w_xyzw"], dtype=np.float64).tolist()
                if "ee_quat_w_xyzw" in proprio
                else None
            ),
            "door_angle_rad": step.object_state.get("door_angle_rad"),
            "door_angular_velocity_rad_s": step.object_state.get(
                "door_angular_velocity_rad_s"
            ),
            "arm_joint_pos_rad": arm_values("joint_pos"),
            "arm_joint_vel_rad_s": arm_values("joint_vel"),
            "arm_joint_target_rad": arm_values("joint_pos_target"),
        }

    if peak_tick is None:
        window = []
        peak = None
    else:
        start = max(0, peak_tick - window_radius)
        stop = min(len(episode.steps), peak_tick + window_radius + 1)
        window = [sample(tick) for tick in range(start, stop)]
        peak = window[peak_tick - start]
    return {
        "all_forces_finite": bool(finite.all()),
        "min_force_n": None if minimum_tick is None else float(forces[minimum_tick]),
        "min_force_tick": minimum_tick,
        "max_force_n": None if peak_tick is None else float(forces[peak_tick]),
        "max_force_tick": peak_tick,
        "force_limit_n": limit,
        "upper_force_gate_passed": bool(finite.all() and not over_limit),
        "non_negative_force_gate_passed": bool(finite.all() and not negative_ticks),
        "force_admission_passed": bool(
            finite.all() and not negative_ticks and not over_limit
        ),
        "non_finite_force_ticks": non_finite_ticks,
        "negative_force_ticks": negative_ticks,
        "ticks_over_limit": over_limit,
        "one_tick_over_limit": len(over_limit) == 1,
        "sustained_over_limit": sustained,
        "peak": peak,
        "window": window,
        "ik_clamp_telemetry": episode.extras.get("ik_clamp_telemetry"),
    }


def check_alex_episode(
    episode: EpisodeBuffer,
    *,
    max_arm_joint_vel_rad_s: float = MAX_ARM_JOINT_VEL_RAD_S,
    force_warn_n: float = FORCE_WARN_N,
    force_error_n: float | None = None,
    limit_margin_rad: float = LIMIT_MARGIN_RAD,
    limit_error_rad: float = LIMIT_ERROR_RAD,
    settle_ticks: int = SETTLE_TICKS,
) -> SanityResult:
    """Validate one force-sensing episode's recorded joint/contact data.

    Errors (gate-failing): non-finite joint state or targets, targets past the
    recorded position limits by more than ``limit_error_rad``, post-settle
    joint velocities above the recorded sim velocity limits, wrong contact
    source. Warnings: target overshoot in the (``limit_margin_rad``,
    ``limit_error_rad``] band (unclamped diff-IK drift while the drive sits at
    the limit), post-settle arm joint velocity above
    ``max_arm_joint_vel_rad_s``, sensed contact force above ``force_warn_n``.
    When ``force_error_n`` is provided, exceeding it is a hard admission error;
    calibration and dataset writers use the unchanged 200 N threshold this way.
    Velocity checks skip the first ``settle_ticks`` ticks (reset transient of
    the position-held joints); finiteness and target-limit checks cover every
    tick.
    """
    result = SanityResult()
    if not episode.steps:
        result.errors.append("episode has no recorded steps")
        return result
    label = f"episode {episode.meta.episode_id[:8]} (seed {episode.meta.seed})"

    missing = [key for key in _PROPRIO_KEYS if key not in episode.steps[0].proprio]
    if missing:
        result.errors.append(f"{label}: proprio is missing {missing}")
        return result

    tables = {
        key: episode.stacked(lambda s, k=key: s.proprio[k]) for key in _PROPRIO_KEYS
    }  # each (N, J)
    for key, values in tables.items():
        if not np.isfinite(values).all():
            ticks = np.nonzero(~np.isfinite(values).all(axis=1))[0]
            result.errors.append(
                f"{label}: non-finite {key} at ticks {ticks[:5].tolist()}"
                + ("..." if ticks.size > 5 else "")
            )
    if result.errors:
        return result

    joint_names = [str(n) for n in episode.extras.get("joint_names", [])]

    def joint_label(j: int) -> str:
        return joint_names[j] if j < len(joint_names) else f"joint[{j}]"

    pos_limits = episode.extras.get("joint_pos_limits")
    if pos_limits is not None:
        limits = np.asarray(pos_limits, dtype=np.float64)  # (J, 2)
        lower, upper = limits[:, 0], limits[:, 1]
        finite = np.isfinite(lower) & np.isfinite(upper)
        targets = tables["joint_pos_target"]
        overshoot = np.maximum(lower - targets, targets - upper)  # (N, J)
        overshoot[:, ~finite] = -np.inf
        worst_per_joint = overshoot.max(axis=0)
        for j in np.nonzero(worst_per_joint > limit_margin_rad)[0]:
            worst = float(worst_per_joint[j])
            message = (
                f"{label}: joint target for {joint_label(j)} exceeds its position "
                f"limits [{lower[j]:.3f}, {upper[j]:.3f}] rad by {worst:.4f} rad"
            )
            if worst > limit_error_rad:
                result.errors.append(message)
            else:
                result.warnings.append(message + " (unclamped diff-IK drift at the limit)")

    settled_vel = tables["joint_vel"][settle_ticks:]
    vel_limits = episode.extras.get("joint_vel_limits")
    if vel_limits is not None and settled_vel.size:
        limits = np.asarray(vel_limits, dtype=np.float64).reshape(-1)  # (J,)
        speeds = np.abs(settled_vel)
        finite = np.isfinite(limits)
        violated = (speeds > limits) & finite
        for j in np.nonzero(violated.any(axis=0))[0]:
            result.errors.append(
                f"{label}: {joint_label(j)} velocity peaked at "
                f"{float(speeds[:, j].max()):.2f} rad/s after settle, above its "
                f"sim limit {limits[j]:.2f} rad/s"
            )

    arm_ids = [int(i) for i in episode.extras.get("arm_joint_ids", [])]
    if arm_ids and settled_vel.size:
        arm_speeds = np.abs(settled_vel[:, arm_ids])
        peak = float(arm_speeds.max())
        if peak > max_arm_joint_vel_rad_s:
            j = arm_ids[int(np.unravel_index(np.argmax(arm_speeds), arm_speeds.shape)[1])]
            result.warnings.append(
                f"{label}: arm joint {joint_label(j)} reached {peak:.2f} rad/s "
                f"after settle (warn threshold {max_arm_joint_vel_rad_s:.2f} rad/s)"
            )

    source = str(episode.steps[0].contact.get("source", ""))
    if source != CONTACT_SOURCE_FORCE:
        result.errors.append(
            f"{label}: contact source must be {CONTACT_SOURCE_FORCE!r}, got {source!r}"
        )

    force_limit = force_warn_n if force_error_n is None else force_error_n
    force_evidence = contact_force_diagnostics(episode, force_limit_n=force_limit)
    if not force_evidence["all_forces_finite"]:
        result.errors.append(f"{label}: non-finite contact force values")
    if force_evidence["negative_force_ticks"]:
        result.errors.append(
            f"{label}: contact force magnitude must be non-negative: "
            f"{force_evidence['min_force_n']:.1f} N at tick "
            f"{force_evidence['min_force_tick']} "
            f"(negative ticks {force_evidence['negative_force_ticks'][:5]})"
        )
    if force_error_n is not None and force_evidence["ticks_over_limit"]:
        result.errors.append(
            f"{label}: contact force exceeded the {force_error_n:.0f} N force admission "
            f"limit: {force_evidence['max_force_n']:.1f} N at tick "
            f"{force_evidence['max_force_tick']}"
        )
    elif (
        force_evidence["max_force_n"] is not None
        and force_evidence["max_force_n"] > force_warn_n
    ):
        result.warnings.append(
            f"{label}: contact force spiked to {force_evidence['max_force_n']:.1f} N at tick "
            f"{force_evidence['max_force_tick']} (warn threshold {force_warn_n:.0f} N)"
        )

    return result


__all__ = [
    "FORCE_DATASET_LIMIT_N",
    "FORCE_WARN_N",
    "LIMIT_ERROR_RAD",
    "LIMIT_MARGIN_RAD",
    "MAX_ARM_JOINT_VEL_RAD_S",
    "SETTLE_TICKS",
    "SanityResult",
    "check_alex_episode",
    "contact_force_diagnostics",
]
