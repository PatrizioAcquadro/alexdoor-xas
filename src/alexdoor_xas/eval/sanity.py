"""Joint and contact validation for recorded Alex V2 episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alexdoor_xas.recording import EpisodeBuffer

CONTACT_SOURCE_FORCE = "force_sensor+geometric"
FORCE_DATASET_LIMIT_N = 200.0

_MAX_ARM_JOINT_VEL_RAD_S = 4.0
_SETTLE_TICKS = 30
_LIMIT_MARGIN_RAD = 0.01
_LIMIT_ERROR_RAD = 0.1
_PROPRIO_KEYS = ("joint_pos", "joint_vel", "joint_pos_target")


@dataclass
class SanityResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def contact_force_summary(episode: EpisodeBuffer) -> dict[str, Any]:
    """Return compact force-admission evidence."""
    forces = np.asarray([_force_value(step.contact) for step in episode.steps], dtype=np.float64)
    finite = np.isfinite(forces)
    finite_ticks = np.flatnonzero(finite)
    peak_tick = (
        int(finite_ticks[np.argmax(forces[finite_ticks])]) if finite_ticks.size else None
    )

    terminal_raw = episode.extras.get("terminal_contact")
    terminal = None
    if terminal_raw is not None:
        force_n = _force_value(terminal_raw)
        terminal_finite = bool(np.isfinite(force_n))
        terminal = {
            "force_n": force_n,
            "sensed": terminal_raw.get("sensed"),
            "t_s": terminal_raw.get("t"),
            "finite": terminal_finite,
            "within_limit": bool(
                terminal_finite and 0.0 <= force_n <= FORCE_DATASET_LIMIT_N
            ),
        }

    return {
        "force_limit_n": FORCE_DATASET_LIMIT_N,
        "max_force_n": None if peak_tick is None else float(forces[peak_tick]),
        "max_force_tick": peak_tick,
        "non_finite_ticks": np.flatnonzero(~finite).astype(int).tolist(),
        "negative_ticks": np.flatnonzero(finite & (forces < 0.0)).astype(int).tolist(),
        "over_limit_ticks": np.flatnonzero(
            finite & (forces > FORCE_DATASET_LIMIT_N)
        ).astype(int).tolist(),
        "terminal": terminal,
    }


def check_alex_episode(episode: EpisodeBuffer) -> SanityResult:
    """Validate recorded Alex V2 joint and contact data."""
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
        key: episode.stacked(lambda step, name=key: step.proprio[name])
        for key in _PROPRIO_KEYS
    }
    for key, values in tables.items():
        if not np.isfinite(values).all():
            ticks = np.nonzero(~np.isfinite(values).all(axis=1))[0]
            suffix = "..." if ticks.size > 5 else ""
            result.errors.append(
                f"{label}: non-finite {key} at ticks {ticks[:5].tolist()}{suffix}"
            )
    if result.errors:
        return result

    joint_names = [str(name) for name in episode.extras.get("joint_names", [])]

    def joint_label(index: int) -> str:
        return joint_names[index] if index < len(joint_names) else f"joint[{index}]"

    pos_limits = episode.extras.get("joint_pos_limits")
    if pos_limits is not None:
        limits = np.asarray(pos_limits, dtype=np.float64)
        lower, upper = limits[:, 0], limits[:, 1]
        finite_limits = np.isfinite(lower) & np.isfinite(upper)
        targets = tables["joint_pos_target"]
        overshoot = np.maximum(lower - targets, targets - upper)
        overshoot[:, ~finite_limits] = -np.inf
        for joint in np.nonzero(overshoot.max(axis=0) > _LIMIT_MARGIN_RAD)[0]:
            worst = float(overshoot[:, joint].max())
            message = (
                f"{label}: joint target for {joint_label(joint)} exceeds its position "
                f"limits [{lower[joint]:.3f}, {upper[joint]:.3f}] rad by {worst:.4f} rad"
            )
            if worst > _LIMIT_ERROR_RAD:
                result.errors.append(message)
            else:
                result.warnings.append(message)

    settled_vel = tables["joint_vel"][_SETTLE_TICKS:]
    vel_limits = episode.extras.get("joint_vel_limits")
    if vel_limits is not None and settled_vel.size:
        limits = np.asarray(vel_limits, dtype=np.float64).reshape(-1)
        speeds = np.abs(settled_vel)
        violated = (speeds > limits) & np.isfinite(limits)
        for joint in np.nonzero(violated.any(axis=0))[0]:
            result.errors.append(
                f"{label}: {joint_label(joint)} velocity peaked at "
                f"{float(speeds[:, joint].max()):.2f} rad/s after settle, above its "
                f"sim limit {limits[joint]:.2f} rad/s"
            )

    arm_ids = [int(index) for index in episode.extras.get("arm_joint_ids", [])]
    if arm_ids and settled_vel.size:
        arm_speeds = np.abs(settled_vel[:, arm_ids])
        peak = float(arm_speeds.max())
        if peak > _MAX_ARM_JOINT_VEL_RAD_S:
            peak_column = int(np.unravel_index(np.argmax(arm_speeds), arm_speeds.shape)[1])
            joint = arm_ids[peak_column]
            result.warnings.append(
                f"{label}: arm joint {joint_label(joint)} reached {peak:.2f} rad/s "
                f"after settle"
            )

    invalid_contacts: list[int] = []
    for tick, step in enumerate(episode.steps):
        contact = step.contact
        if (
            contact.get("source") != CONTACT_SOURCE_FORCE
            or not isinstance(contact.get("sensed"), (bool, np.bool_))
            or "force_n" not in contact
        ):
            invalid_contacts.append(tick)
    if invalid_contacts:
        result.errors.append(
            f"{label}: invalid contact source or {CONTACT_SOURCE_FORCE!r} fields at ticks "
            f"{invalid_contacts[:5]}"
        )

    force = contact_force_summary(episode)
    if force["non_finite_ticks"]:
        result.errors.append(
            f"{label}: non-finite contact force at ticks {force['non_finite_ticks'][:5]}"
        )
    if force["negative_ticks"]:
        result.errors.append(
            f"{label}: contact force magnitude is negative at ticks "
            f"{force['negative_ticks'][:5]}"
        )
    if force["over_limit_ticks"]:
        result.errors.append(
            f"{label}: contact force exceeded the {FORCE_DATASET_LIMIT_N:.0f} N "
            f"force admission limit at ticks {force['over_limit_ticks'][:5]}"
        )

    terminal = force["terminal"]
    if terminal is not None:
        terminal_force = terminal["force_n"]
        if not terminal["finite"]:
            result.errors.append(f"{label}: terminal contact force is non-finite")
        elif terminal_force < 0.0:
            result.errors.append(f"{label}: terminal contact force magnitude is negative")
        elif not terminal["within_limit"]:
            result.errors.append(
                f"{label}: terminal contact force exceeded the "
                f"{FORCE_DATASET_LIMIT_N:.0f} N force admission limit"
            )

    return result


def _force_value(contact: dict[str, Any]) -> float:
    try:
        return float(contact["force_n"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
