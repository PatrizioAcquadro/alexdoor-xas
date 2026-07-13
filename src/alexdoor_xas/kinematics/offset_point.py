"""Rigid offset-pose and point-Jacobian math implemented with pure Torch.

The functions in this module use meters, world-frame spatial Jacobians ordered
as ``(linear, angular)``, and quaternions ordered as ``(x, y, z, w)``.  They do
not import Isaac Sim or Isaac Lab, which keeps the math independently testable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_QUATERNION_NORM_ATOL = 1.0e-5


def compose_offset_pose_xyzw(
    link_position_w: torch.Tensor,
    link_orientation_w_xyzw: torch.Tensor,
    offset_position_link: torch.Tensor,
    offset_orientation_link_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a link pose with a rigid local offset.

    Args:
        link_position_w: Link-origin position in world, shape ``(..., 3)``.
        link_orientation_w_xyzw: Rotation ``world <- link``, shape ``(..., 4)``.
        offset_position_link: Offset-frame origin in link coordinates, shape
            ``(..., 3)``.
        offset_orientation_link_xyzw: Rotation ``link <- offset``, shape
            ``(..., 4)``.

    Leading dimensions follow normal Torch broadcasting.  All tensors must be
    finite floating-point tensors on the same device and with the same dtype;
    both quaternion inputs must already be unit length.
    """

    values = (
        ("link_position_w", link_position_w, 3),
        ("link_orientation_w_xyzw", link_orientation_w_xyzw, 4),
        ("offset_position_link", offset_position_link, 3),
        ("offset_orientation_link_xyzw", offset_orientation_link_xyzw, 4),
    )
    _validate_tensors(values)
    _validate_unit_quaternion(link_orientation_w_xyzw, "link_orientation_w_xyzw")
    _validate_unit_quaternion(offset_orientation_link_xyzw, "offset_orientation_link_xyzw")
    batch_shape = _broadcast_batch_shape(
        tuple((name, tensor, 1) for name, tensor, _ in values)
    )
    link_position_w = _expand_batch(link_position_w, batch_shape, 1)
    link_orientation_w_xyzw = _expand_batch(link_orientation_w_xyzw, batch_shape, 1)
    offset_position_link = _expand_batch(offset_position_link, batch_shape, 1)
    offset_orientation_link_xyzw = _expand_batch(
        offset_orientation_link_xyzw, batch_shape, 1
    )

    offset_position_w = _rotate_vector_xyzw(
        link_orientation_w_xyzw, offset_position_link
    )
    position_w = link_position_w + offset_position_w
    orientation_w_xyzw = _quaternion_multiply_xyzw(
        link_orientation_w_xyzw, offset_orientation_link_xyzw
    )
    return position_w, orientation_w_xyzw


def link_jacobian_to_point(
    link_jacobian_w: torch.Tensor,
    link_orientation_w_xyzw: torch.Tensor,
    offset_position_link: torch.Tensor,
) -> torch.Tensor:
    """Move a world-frame spatial Jacobian from a link origin to a rigid point.

    ``link_jacobian_w`` has shape ``(..., 6, N)`` and row order
    ``(linear_xyz, angular_xyz)``.  The local offset is rotated into world as
    ``r_world`` and the returned linear block is
    ``Jv - skew(r_world) @ Jw``.  The angular block is unchanged.
    """

    values = (
        ("link_jacobian_w", link_jacobian_w, 2),
        ("link_orientation_w_xyzw", link_orientation_w_xyzw, 1),
        ("offset_position_link", offset_position_link, 1),
    )
    _validate_tensor(link_jacobian_w, "link_jacobian_w")
    if link_jacobian_w.ndim < 2 or link_jacobian_w.shape[-2] != 6:
        raise ValueError(
            "link_jacobian_w must have shape (..., 6, N), "
            f"got {tuple(link_jacobian_w.shape)}"
        )
    if link_jacobian_w.shape[-1] <= 0:
        raise ValueError("link_jacobian_w must have at least one joint column")
    _validate_tensors(
        (
            ("link_orientation_w_xyzw", link_orientation_w_xyzw, 4),
            ("offset_position_link", offset_position_link, 3),
        ),
        reference=("link_jacobian_w", link_jacobian_w),
    )
    _validate_unit_quaternion(link_orientation_w_xyzw, "link_orientation_w_xyzw")
    batch_shape = _broadcast_batch_shape(values)
    joint_count = link_jacobian_w.shape[-1]
    link_jacobian_w = _expand_batch(link_jacobian_w, batch_shape, 2)
    link_orientation_w_xyzw = _expand_batch(link_orientation_w_xyzw, batch_shape, 1)
    offset_position_link = _expand_batch(offset_position_link, batch_shape, 1)

    offset_position_w = _rotate_vector_xyzw(
        link_orientation_w_xyzw, offset_position_link
    )
    skew_offset_w = _skew_symmetric(offset_position_w)
    linear = (
        link_jacobian_w[..., :3, :]
        - skew_offset_w @ link_jacobian_w[..., 3:, :]
    )
    angular = link_jacobian_w[..., 3:, :]
    result = torch.cat((linear, angular), dim=-2)
    assert result.shape == batch_shape + (6, joint_count)
    return result


def world_vector_to_link_xyzw(
    link_orientation_w_xyzw: torch.Tensor,
    vector_w: torch.Tensor,
) -> torch.Tensor:
    """Re-express a world-frame free vector in a link frame.

    ``link_orientation_w_xyzw`` is the live ``world <- link`` orientation.
    The inverse rotation is therefore its quaternion conjugate.  Leading
    dimensions follow normal Torch broadcasting.
    """

    values = (
        ("link_orientation_w_xyzw", link_orientation_w_xyzw, 4),
        ("vector_w", vector_w, 3),
    )
    _validate_tensors(values)
    _validate_unit_quaternion(link_orientation_w_xyzw, "link_orientation_w_xyzw")
    batch_shape = _broadcast_batch_shape(
        tuple((name, tensor, 1) for name, tensor, _ in values)
    )
    link_orientation_w_xyzw = _expand_batch(
        link_orientation_w_xyzw, batch_shape, 1
    )
    vector_w = _expand_batch(vector_w, batch_shape, 1)
    inverse_orientation_xyzw = torch.cat(
        (-link_orientation_w_xyzw[..., :3], link_orientation_w_xyzw[..., 3:]),
        dim=-1,
    )
    return _rotate_vector_xyzw(inverse_orientation_xyzw, vector_w)


def _validate_tensors(
    values: Sequence[tuple[str, torch.Tensor, int]],
    *,
    reference: tuple[str, torch.Tensor] | None = None,
) -> None:
    first_name, first_tensor = reference or (values[0][0], values[0][1])
    _validate_tensor(first_tensor, first_name)
    for name, tensor, trailing_size in values:
        _validate_tensor(tensor, name)
        if tensor.ndim < 1 or tensor.shape[-1] != trailing_size:
            raise ValueError(
                f"{name} must have shape (..., {trailing_size}), got {tuple(tensor.shape)}"
            )
        if tensor.dtype != first_tensor.dtype:
            raise TypeError(
                f"{name} dtype {tensor.dtype} does not match {first_name} dtype "
                f"{first_tensor.dtype}"
            )
        if tensor.device != first_tensor.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {first_name} device "
                f"{first_tensor.device}"
            )


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_unit_quaternion(quaternion: torch.Tensor, name: str) -> None:
    norms = torch.linalg.vector_norm(quaternion, dim=-1)
    errors = torch.abs(norms - 1.0)
    if bool(torch.any(errors > _QUATERNION_NORM_ATOL)):
        max_error = float(errors.max().detach().cpu())
        raise ValueError(
            f"{name} must contain unit XYZW quaternions; maximum norm error is "
            f"{max_error:.3e}"
        )


def _broadcast_batch_shape(values: Sequence[tuple[str, torch.Tensor, int]]) -> torch.Size:
    shapes = [tensor.shape[:-trailing_dims] for _, tensor, trailing_dims in values]
    try:
        return torch.broadcast_shapes(*shapes)
    except RuntimeError as error:
        formatted = ", ".join(
            f"{name}={tuple(shape)}" for (name, _, _), shape in zip(values, shapes, strict=True)
        )
        raise ValueError(f"tensor batch dimensions are not broadcastable: {formatted}") from error


def _expand_batch(
    tensor: torch.Tensor, batch_shape: torch.Size, trailing_dims: int
) -> torch.Tensor:
    return tensor.expand(batch_shape + tensor.shape[-trailing_dims:])


def _rotate_vector_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector_part = quaternion[..., :3]
    scalar_part = quaternion[..., 3:]
    first_cross = torch.linalg.cross(vector_part, vector, dim=-1)
    second_cross = torch.linalg.cross(vector_part, first_cross, dim=-1)
    return vector + 2.0 * (scalar_part * first_cross + second_cross)


def _quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_vector, left_scalar = left[..., :3], left[..., 3:]
    right_vector, right_scalar = right[..., :3], right[..., 3:]
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + torch.linalg.cross(left_vector, right_vector, dim=-1)
    )
    scalar = left_scalar * right_scalar - (left_vector * right_vector).sum(
        dim=-1, keepdim=True
    )
    return torch.cat((vector, scalar), dim=-1)


def _skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


__all__ = [
    "compose_offset_pose_xyzw",
    "link_jacobian_to_point",
    "world_vector_to_link_xyzw",
]
