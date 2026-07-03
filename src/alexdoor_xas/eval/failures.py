"""Deterministic failure taxonomy for scripted door-push episodes (Phase 2).

The vocabulary for the schema's ``outcome.failure_label``:

- ``non_finite_state`` — the sim produced NaN/inf and the env aborted the episode.
- ``phase_timeout_<phase>`` — the controller exhausted a phase's tick budget
  (e.g. ``phase_timeout_contact``: it never inferred contact).
- ``env_truncated_before_completion`` — the episode horizon ended before the
  controller finished its phase sequence.
- ``insufficient_final_angle`` — the controller completed all phases but the
  door ended below the success threshold (e.g. it swung back after release).

A successful episode has ``failure_label = None``.
"""

from __future__ import annotations

import math

from alexdoor_xas.policies.scripted import DoorPushPhase

FAILURE_LABELS: tuple[str, ...] = (
    "non_finite_state",
    *(f"phase_timeout_{phase}" for phase in DoorPushPhase if phase is not DoorPushPhase.DONE),
    "env_truncated_before_completion",
    "insufficient_final_angle",
)


def label_episode(
    *,
    final_angle_rad: float,
    success_angle_rad: float,
    controller_done: bool,
    timed_out: bool,
    last_phase: str,
    notes: str = "",
) -> str | None:
    """Return the failure label for one finished episode (``None`` = success)."""
    if not math.isfinite(final_angle_rad) or "env.step failed" in notes:
        return "non_finite_state"
    if final_angle_rad >= success_angle_rad:
        return None
    if timed_out:
        return f"phase_timeout_{last_phase}"
    if not controller_done:
        return "env_truncated_before_completion"
    return "insufficient_final_angle"


__all__ = ["FAILURE_LABELS", "label_episode"]
