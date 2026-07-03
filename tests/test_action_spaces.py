"""Pure tests for action-space tags, structs, and door-frame math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alexdoor_xas.action import frames, spaces


def test_action_space_tags_are_canonical() -> None:
    assert spaces.ALL_ACTION_SPACES == (
        "A1_joint_delta",
        "A2_ee_delta",
        "A3_obj_rel_ee_delta",
        "A4_obj_centric_chunk",
    )
    assert spaces.EXPORTED_ACTION_SPACES == (
        "A2_ee_delta",
        "A3_obj_rel_ee_delta",
        "A4_obj_centric_chunk",
    )
    assert spaces.A1_JOINT_DELTA not in spaces.EXPORTED_ACTION_SPACES
    assert spaces.EE_DELTA_DIM == 6


def test_object_centric_chunk_round_trips_through_dict() -> None:
    chunk = spaces.ObjectCentricChunk(
        phase="push",
        contact_target_panel=(0.086, 0.664, 0.0),
        motion_hinge_delta_rad=math.radians(50.0),
        duration_ticks=120,
    )
    assert spaces.ObjectCentricChunk.from_dict(chunk.to_dict()) == chunk


def test_quat_to_rot_matrix_identity_and_yaw() -> None:
    # Quaternions are (x, y, z, w) — the Isaac Lab 3.0 data layout.
    identity = frames.quat_to_rot_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

    half = math.pi / 4.0  # 90 deg yaw
    yaw90 = frames.quat_to_rot_matrix(np.array([0.0, 0.0, math.sin(half), math.cos(half)]))
    np.testing.assert_allclose(yaw90, frames.rot_z(math.pi / 2.0), atol=1e-12)


def test_quat_to_rot_matrix_rejects_degenerate_quaternion() -> None:
    with pytest.raises(ValueError, match="finite and non-zero"):
        frames.quat_to_rot_matrix(np.zeros(4))


def test_panel_frame_rotates_points_about_hinge_axis() -> None:
    door = frames.ObjectFrame(origin=np.zeros(3), rot=np.eye(3))
    panel = frames.panel_frame(door, math.pi / 2.0)
    # A point one meter along the panel (+Y) swings to -X when the door is open 90 deg.
    np.testing.assert_allclose(
        panel.point_to_world(np.array([0.0, 1.0, 0.0])), [-1.0, 0.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_world_frame_delta_round_trip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=4)
    frame = frames.ObjectFrame(origin=rng.normal(size=3), rot=frames.quat_to_rot_matrix(quat))
    delta_world = rng.normal(size=spaces.EE_DELTA_DIM)

    delta_frame = frames.world_delta_to_frame(delta_world, frame)
    np.testing.assert_allclose(
        frames.frame_delta_to_world(delta_frame, frame), delta_world, atol=1e-12
    )
    # Norms are preserved: the conversion is a pure rotation of both 3-vectors.
    assert np.linalg.norm(delta_frame[:3]) == pytest.approx(np.linalg.norm(delta_world[:3]))
    assert np.linalg.norm(delta_frame[3:]) == pytest.approx(np.linalg.norm(delta_world[3:]))


def test_delta_conversion_rejects_wrong_shape() -> None:
    frame = frames.ObjectFrame(origin=np.zeros(3), rot=np.eye(3))
    with pytest.raises(ValueError, match="EE delta must have shape"):
        frames.world_delta_to_frame(np.zeros(3), frame)


def test_point_round_trip_through_frame() -> None:
    rng = np.random.default_rng(7)
    frame = frames.ObjectFrame(
        origin=rng.normal(size=3), rot=frames.rot_z(rng.uniform(-math.pi, math.pi))
    )
    point = rng.normal(size=3)
    np.testing.assert_allclose(
        frame.point_to_world(frame.point_from_world(point)), point, atol=1e-12
    )
