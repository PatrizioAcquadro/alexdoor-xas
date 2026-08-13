"""Physical invariants for the Alex V2 tool-point Jacobian."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import combine_frame_transforms, quat_mul

from alexdoor_xas.kinematics.point_jacobian import link_jacobian_to_point


def _yaw_quaternion(angle_rad: float) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0)]],
        dtype=torch.float64,
    )


def test_point_jacobian_matches_rotated_lever_arm() -> None:
    jacobian = torch.zeros((1, 6, 3), dtype=torch.float64)
    jacobian[:, 3:, :] = torch.eye(3, dtype=torch.float64)

    result = link_jacobian_to_point(
        jacobian,
        _yaw_quaternion(math.pi / 2.0),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )

    expected_linear = torch.tensor(
        [[[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result[:, :3], expected_linear, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(result[:, 3:], jacobian[:, 3:])


def test_point_jacobian_matches_finite_difference_velocity() -> None:
    link_position = torch.tensor([[0.3, -0.4, 1.2]], dtype=torch.float64)
    axis = torch.tensor([[0.2, -0.5, 0.3]], dtype=torch.float64)
    axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angle = 0.7
    link_orientation = torch.cat(
        (axis * math.sin(angle / 2.0), torch.tensor([[math.cos(angle / 2.0)]])), dim=-1
    )
    offset = torch.tensor([[0.12, -0.08, 0.05]], dtype=torch.float64)
    jacobian = torch.tensor(
        [
            [
                [0.2, -0.1, 0.4, 0.0],
                [0.3, 0.5, -0.2, 0.1],
                [-0.4, 0.2, 0.3, -0.1],
                [0.7, -0.2, 0.0, 0.4],
                [-0.1, 0.6, -0.5, 0.2],
                [0.3, 0.1, 0.8, -0.4],
            ]
        ],
        dtype=torch.float64,
    )
    point_jacobian = link_jacobian_to_point(jacobian, link_orientation, offset)
    point_position, _ = combine_frame_transforms(link_position, link_orientation, offset)
    dt = 1.0e-7

    for column in range(jacobian.shape[-1]):
        angular_velocity = jacobian[:, 3:, column]
        speed = torch.linalg.vector_norm(angular_velocity, dim=-1, keepdim=True)
        half_angle = speed * dt / 2.0
        delta_orientation = torch.cat(
            (angular_velocity / speed * torch.sin(half_angle), torch.cos(half_angle)), dim=-1
        )
        next_orientation = quat_mul(delta_orientation, link_orientation)
        next_position = link_position + jacobian[:, :3, column] * dt
        next_point_position, _ = combine_frame_transforms(next_position, next_orientation, offset)
        velocity = (next_point_position - point_position) / dt
        torch.testing.assert_close(
            velocity,
            point_jacobian[:, :3, column],
            atol=1e-7,
            rtol=1e-7,
        )

    torch.testing.assert_close(point_jacobian[:, 3:], jacobian[:, 3:])
