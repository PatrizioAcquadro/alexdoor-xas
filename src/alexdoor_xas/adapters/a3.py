"""A3 adapter: object-relative (door-frame) EE delta -> A2 execution.

Adapter-v1 for ``A3_obj_rel_ee_delta``: validates that the object frame is
available and trustworthy, re-expresses the door-frame delta in world
coordinates (`action/frames.frame_delta_to_world` — the frozen A3 -> A2
conversion), and delegates execution to the :class:`A2Adapter`.

Transform assumption (recorded in every decision's checks): the door frame is
the hinge-anchored ``Doorframe`` body frame, read from the USD stage at reset
and **static** for the whole episode — live articulation pose reads return
zeros for the referenced door in this Isaac Lab build (CLAUDE.md gotchas).
Push/pull semantics survive the transform: the delta is rotated, never
mirrored, so a door-frame ``-x`` push stays a push on the panel's +X face.
"""

from __future__ import annotations

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame, frame_delta_to_world
from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .a2 import A2Adapter
from .base import AdapterDecision, AdapterStatus, StepContext

_ROT_ORTHONORMAL_TOL = 1e-5


def validate_object_frame(frame: ObjectFrame | None) -> str:
    """Return the failure reason if the object frame is missing/corrupt, else ''."""
    if frame is None:
        return "object frame is unavailable (no door frame in the step context)"
    origin = np.asarray(frame.origin, dtype=np.float64).reshape(-1)
    rot = np.asarray(frame.rot, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return f"object frame origin is invalid: {origin}"
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        return "object frame rotation is non-finite or misshaped"
    if not np.allclose(rot @ rot.T, np.eye(3), atol=_ROT_ORTHONORMAL_TOL):
        return "object frame rotation is not orthonormal"
    if not np.isclose(np.linalg.det(rot), 1.0, atol=_ROT_ORTHONORMAL_TOL):
        return "object frame rotation is not proper (determinant must be +1)"
    return ""


class A3Adapter:
    """Convert door-frame EE deltas to world frame and execute through A2."""

    def __init__(self, a2: A2Adapter):
        self.a2 = a2

    @property
    def log(self):
        return self.a2.log

    def process(self, delta_door_frame, ctx: StepContext) -> tuple[np.ndarray, AdapterDecision]:
        """Adapt one 6-dim door-frame EE delta; returns (applied_world, decision)."""
        requested = np.asarray(delta_door_frame, dtype=np.float64).reshape(-1)
        frame_reason = validate_object_frame(ctx.door_frame)
        hinge_ok = bool(np.isfinite(ctx.hinge_angle_rad))
        if frame_reason or not hinge_ok:
            decision = self.log.record(
                AdapterDecision(
                    status=AdapterStatus.REJECTED,
                    reason=frame_reason or "hinge angle is non-finite",
                    checks={"object_frame_trusted": not frame_reason, "hinge_readable": hinge_ok},
                    requested=requested,
                    applied=np.zeros(EE_DELTA_DIM),
                )
            )
            return np.zeros(EE_DELTA_DIM), decision
        if requested.shape != (EE_DELTA_DIM,) or not np.isfinite(requested).all():
            # Let A2 produce the canonical shape/finiteness rejection.
            return self.a2.process(requested, ctx)

        delta_world = frame_delta_to_world(requested, ctx.door_frame)
        applied, a2_decision = self.a2.process(delta_world, ctx)
        # Annotate the recorded decision with the A3 view of the command and
        # the transform assumption (static hinge-anchored door frame).
        checks = dict(a2_decision.checks)
        checks["object_frame_trusted"] = True
        checks["door_frame_static_stage_read"] = True
        annotated = AdapterDecision(
            status=a2_decision.status,
            reason=a2_decision.reason,
            checks=checks,
            warnings=a2_decision.warnings,
            warning_records=a2_decision.warning_records,
            requested=requested,
            applied=applied,
        )
        self.log.decisions[-1] = annotated
        return applied, annotated


__all__ = ["A3Adapter", "validate_object_frame"]
