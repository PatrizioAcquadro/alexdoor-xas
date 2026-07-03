"""Canonical action-space tags and structs (docs/action_spaces.md, operational form).

The tags are the single source of dispatch for episode metadata, dataset layout,
and (later) action-space-conditioned policies. Phase 2 exports A2/A3/A4 from the
scripted proxy-end-effector baseline; A1 stays a documented placeholder because
the proxy has no joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

A1_JOINT_DELTA = "A1_joint_delta"
A2_EE_DELTA = "A2_ee_delta"
A3_OBJ_REL_EE_DELTA = "A3_obj_rel_ee_delta"
A4_OBJ_CENTRIC_CHUNK = "A4_obj_centric_chunk"

ALL_ACTION_SPACES: tuple[str, ...] = (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)

# Spaces the data engine exports for every robot. A1 is additionally exported
# when per-tick joint targets were recorded (Alex episodes, since Phase 2.5);
# the proxy sphere has no joints, so proxy runs stay A1-less (docs/action_spaces.md).
EXPORTED_ACTION_SPACES: tuple[str, ...] = (
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)

# A2/A3 per-step action layout: (dx, dy, dz, drx, dry, drz) for one end-effector.
EE_DELTA_DIM = 6


@dataclass(frozen=True)
class ObjectCentricChunk:
    """One A4 chunk: what to do to the object during one controller phase.

    ``contact_target_panel`` is a point in the door *panel* frame (moves with the
    door), so the chunk stays valid under any door pose. ``motion_hinge_delta_rad``
    is the intended change of the hinge angle over the chunk (0 for non-push
    phases). ``duration_ticks`` is filled when the phase exits.
    """

    phase: str
    contact_target_panel: tuple[float, float, float]
    motion_hinge_delta_rad: float
    duration_ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "contact_target_panel": list(self.contact_target_panel),
            "motion_hinge_delta_rad": self.motion_hinge_delta_rad,
            "duration_ticks": self.duration_ticks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectCentricChunk:
        return cls(
            phase=str(data["phase"]),
            contact_target_panel=tuple(float(v) for v in data["contact_target_panel"]),
            motion_hinge_delta_rad=float(data["motion_hinge_delta_rad"]),
            duration_ticks=int(data["duration_ticks"]),
        )


@dataclass
class ChunkLog:
    """Ordered A4 chunks emitted by one episode of the scripted controller."""

    chunks: list[ObjectCentricChunk] = field(default_factory=list)

    def to_list(self) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.chunks]


__all__ = [
    "A1_JOINT_DELTA",
    "A2_EE_DELTA",
    "A3_OBJ_REL_EE_DELTA",
    "A4_OBJ_CENTRIC_CHUNK",
    "ALL_ACTION_SPACES",
    "EE_DELTA_DIM",
    "EXPORTED_ACTION_SPACES",
    "ChunkLog",
    "ObjectCentricChunk",
]
