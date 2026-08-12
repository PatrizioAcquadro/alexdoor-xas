"""Canonical action-space tags and structs (see the action-representation wiki page).

The tags are the single source of dispatch for episode metadata, dataset layout,
and action-space-conditioned policies. Alex V2 exports all four representations
from one recorded episode set.
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

# Spaces exported without requiring recorded joint targets. A1 is additionally
# exported whenever per-tick joint targets are present.
EXPORTED_ACTION_SPACES: tuple[str, ...] = (
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)

# A2/A3 per-step action layout: (dx, dy, dz, drx, dry, drz) for one end-effector.
EE_DELTA_DIM = 6

A4_PHASE_VOCAB: tuple[str, ...] = (
    "approach",
    "align",
    "pre_contact",
    "contact",
    "push",
    "hold",
    "release",
)
"""Frozen A4 phase vocabulary — the scripted controller's emitting phases
(``policies/scripted.DoorPushPhase`` minus the terminal ``done``). Hardcoded so
A4 consumers (dataset, adapters) never import the scripted policy; a unit test
pins the two."""


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
    "A4_PHASE_VOCAB",
    "ALL_ACTION_SPACES",
    "EE_DELTA_DIM",
    "EXPORTED_ACTION_SPACES",
    "ChunkLog",
    "ObjectCentricChunk",
]
