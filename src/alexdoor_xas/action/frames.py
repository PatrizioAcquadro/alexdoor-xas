"""Door/object frame math for A2 <-> A3 conversion (pure numpy, no Isaac imports).

Frames follow the scene convention (Z-up, meters). Quaternions are ``(x, y, z, w)``
— the layout this Isaac Lab release uses for all articulation/rigid-body data.

The *door frame* is the hinge-anchored frame of the door fixture: its origin is
the ``Doorframe`` body origin (the hinge axis passes through it, see
``assets/door_task.py``) and its axes are the ``Doorframe`` body axes (+Z is the
hinge axis). The *panel frame* is the door frame rotated about +Z by the current
hinge angle — it moves with the door panel, so points expressed in it (e.g. a
push target on the panel) stay valid while the door opens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spaces import EE_DELTA_DIM


def quat_to_rot_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix (world <- body) from an ``(x, y, z, w)`` quaternion."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"quaternion must be finite and non-zero, got {quat}")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix for a right-handed rotation about +Z."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class ObjectFrame:
    """A rigid frame: ``origin`` in world, ``rot`` mapping frame -> world vectors."""

    origin: np.ndarray  # (3,)
    rot: np.ndarray  # (3, 3), world <- frame

    def point_to_world(self, point_frame: np.ndarray) -> np.ndarray:
        return self.origin + self.rot @ np.asarray(point_frame, dtype=np.float64).reshape(3)

    def point_from_world(self, point_world: np.ndarray) -> np.ndarray:
        return self.rot.T @ (np.asarray(point_world, dtype=np.float64).reshape(3) - self.origin)

    def vector_to_world(self, vector_frame: np.ndarray) -> np.ndarray:
        return self.rot @ np.asarray(vector_frame, dtype=np.float64).reshape(3)

    def vector_from_world(self, vector_world: np.ndarray) -> np.ndarray:
        return self.rot.T @ np.asarray(vector_world, dtype=np.float64).reshape(3)


def door_frame_from_body_pose(
    frame_pos_w: np.ndarray, frame_quat_w_xyzw: np.ndarray
) -> ObjectFrame:
    """Door (hinge-anchored) frame from the ``Doorframe`` body pose."""
    origin = np.asarray(frame_pos_w, dtype=np.float64).reshape(3)
    return ObjectFrame(origin=origin, rot=quat_to_rot_matrix(frame_quat_w_xyzw))


def panel_frame(door_frame: ObjectFrame, hinge_angle_rad: float) -> ObjectFrame:
    """Panel-attached frame: the door frame rotated about the hinge (+Z) axis."""
    return ObjectFrame(origin=door_frame.origin, rot=door_frame.rot @ rot_z(hinge_angle_rad))


def world_delta_to_frame(delta_world: np.ndarray, frame: ObjectFrame) -> np.ndarray:
    """Re-express a 6-dim EE delta ``(dpos, drot)`` from world into ``frame``.

    Both the translational and the (axis-angle) rotational parts are free
    vectors, so each rotates by ``rot.T``. This is the A2 -> A3 conversion.
    """
    delta = _as_ee_delta(delta_world)
    out = np.empty(EE_DELTA_DIM, dtype=np.float64)
    out[:3] = frame.vector_from_world(delta[:3])
    out[3:] = frame.vector_from_world(delta[3:])
    return out


def frame_delta_to_world(delta_frame: np.ndarray, frame: ObjectFrame) -> np.ndarray:
    """Inverse of :func:`world_delta_to_frame` (A3 -> A2)."""
    delta = _as_ee_delta(delta_frame)
    out = np.empty(EE_DELTA_DIM, dtype=np.float64)
    out[:3] = frame.vector_to_world(delta[:3])
    out[3:] = frame.vector_to_world(delta[3:])
    return out


def _as_ee_delta(delta: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.shape != (EE_DELTA_DIM,):
        raise ValueError(f"EE delta must have shape ({EE_DELTA_DIM},), got {arr.shape}")
    return arr


__all__ = [
    "ObjectFrame",
    "door_frame_from_body_pose",
    "frame_delta_to_world",
    "panel_frame",
    "quat_to_rot_matrix",
    "rot_z",
    "world_delta_to_frame",
]
