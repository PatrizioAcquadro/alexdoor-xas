"""Pure Torch tests for rigid offset-pose and point-Jacobian math."""

from __future__ import annotations

import math

import pytest
import torch

from alexdoor_xas.kinematics.offset_point import (
    compose_offset_pose_xyzw,
    link_jacobian_to_point,
    world_vector_to_link_xyzw,
)


def _yaw_quaternion(angle_rad: float) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0)],
        dtype=torch.float64,
    )


def _multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    vector = (
        left[3] * right[:3]
        + right[3] * left[:3]
        + torch.linalg.cross(left[:3], right[:3])
    )
    scalar = left[3] * right[3] - torch.dot(left[:3], right[:3])
    return torch.cat((vector, scalar.reshape(1)))


def _world_delta_quaternion(angular_velocity_w: torch.Tensor, dt: float) -> torch.Tensor:
    speed = torch.linalg.vector_norm(angular_velocity_w)
    if float(speed) == 0.0:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=angular_velocity_w.dtype)
    half_angle = speed * dt / 2.0
    xyz = angular_velocity_w / speed * torch.sin(half_angle)
    return torch.cat((xyz, torch.cos(half_angle).reshape(1)))


def test_compose_offset_pose_applies_translation_and_rotation_in_link_frame() -> None:
    link_position = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    link_orientation = _yaw_quaternion(math.pi / 2.0)
    offset_position = torch.tensor([0.4, 0.0, -0.2], dtype=torch.float64)
    offset_orientation = _yaw_quaternion(math.pi / 2.0)

    position, orientation = compose_offset_pose_xyzw(
        link_position, link_orientation, offset_position, offset_orientation
    )

    torch.testing.assert_close(
        position, torch.tensor([1.0, -1.6, 0.3], dtype=torch.float64)
    )
    torch.testing.assert_close(
        orientation, torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_link_jacobian_to_point_matches_analytic_rotated_lever_arm() -> None:
    jacobian = torch.zeros((6, 3), dtype=torch.float64)
    jacobian[3:, :] = torch.eye(3, dtype=torch.float64)

    point_jacobian = link_jacobian_to_point(
        jacobian,
        _yaw_quaternion(math.pi / 2.0),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )

    expected_linear = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(point_jacobian[:3], expected_linear, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(point_jacobian[3:], jacobian[3:])


def test_point_jacobian_matches_finite_difference_point_velocity() -> None:
    link_position = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64)
    axis = torch.tensor([0.2, -0.5, 0.3], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = 0.7
    link_orientation = torch.cat(
        (axis * math.sin(angle / 2.0), torch.tensor([math.cos(angle / 2.0)]))
    )
    offset_position = torch.tensor([0.12, -0.08, 0.05], dtype=torch.float64)
    offset_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    jacobian = torch.tensor(
        [
            [0.2, -0.1, 0.4, 0.0],
            [0.3, 0.5, -0.2, 0.1],
            [-0.4, 0.2, 0.3, -0.1],
            [0.7, -0.2, 0.0, 0.4],
            [-0.1, 0.6, -0.5, 0.2],
            [0.3, 0.1, 0.8, -0.4],
        ],
        dtype=torch.float64,
    )
    point_jacobian = link_jacobian_to_point(
        jacobian, link_orientation, offset_position
    )
    point_position, _ = compose_offset_pose_xyzw(
        link_position, link_orientation, offset_position, offset_orientation
    )
    dt = 1.0e-7

    for column in range(jacobian.shape[-1]):
        angular_velocity_w = jacobian[3:, column]
        delta_orientation_w = _world_delta_quaternion(angular_velocity_w, dt)
        next_link_orientation = _multiply_xyzw(delta_orientation_w, link_orientation)
        next_link_position = link_position + jacobian[:3, column] * dt
        next_point_position, _ = compose_offset_pose_xyzw(
            next_link_position,
            next_link_orientation,
            offset_position,
            offset_orientation,
        )
        finite_difference_velocity = (next_point_position - point_position) / dt
        torch.testing.assert_close(
            finite_difference_velocity,
            point_jacobian[:3, column],
            atol=1e-7,
            rtol=1e-7,
        )

    torch.testing.assert_close(point_jacobian[3:], jacobian[3:])


def test_functions_broadcast_constant_offset_across_batch() -> None:
    positions = torch.zeros((2, 3), dtype=torch.float32)
    orientations = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32
    )
    offset = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    offset_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    jacobians = torch.zeros((2, 6, 1), dtype=torch.float32)

    point_positions, _ = compose_offset_pose_xyzw(
        positions, orientations, offset, offset_orientation
    )
    point_jacobians = link_jacobian_to_point(jacobians, orientations, offset)

    torch.testing.assert_close(
        point_positions, torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    )
    assert point_jacobians.shape == (2, 6, 1)


def test_world_vector_is_reexpressed_with_inverse_live_link_rotation() -> None:
    orientation_w_link = _yaw_quaternion(math.pi / 2.0)
    contact_normal_w = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)

    contact_normal_link = world_vector_to_link_xyzw(
        orientation_w_link, contact_normal_w
    )

    torch.testing.assert_close(
        contact_normal_link,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_world_vector_helper_broadcasts_and_preserves_vector_norm() -> None:
    orientations = torch.stack(
        (_yaw_quaternion(0.0), _yaw_quaternion(math.pi)), dim=0
    )
    vector_w = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    vectors_link = world_vector_to_link_xyzw(orientations, vector_w)

    torch.testing.assert_close(
        vectors_link,
        torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(vectors_link, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (
            lambda: compose_offset_pose_xyzw(
                torch.zeros(3),
                torch.tensor([0.0, 0.0, 0.0, 2.0]),
                torch.zeros(3),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
            "unit XYZW",
        ),
        (
            lambda: link_jacobian_to_point(
                torch.zeros((3, 6)),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
                torch.zeros(3),
            ),
            r"shape \(\.\.\., 6, N\)",
        ),
        (
            lambda: link_jacobian_to_point(
                torch.zeros((6, 2)),
                torch.tensor([0.0, 0.0, float("nan"), 1.0]),
                torch.zeros(3),
            ),
            "finite",
        ),
        (
            lambda: link_jacobian_to_point(
                torch.zeros((2, 6, 2)),
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                torch.zeros(3),
            ),
            "not broadcastable",
        ),
    ],
)
def test_functions_reject_invalid_contracts(operation, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        operation()
