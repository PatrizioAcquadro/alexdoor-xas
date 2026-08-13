"""Tool-point Jacobian math for XYZW Isaac Lab tensors."""

import torch
from isaaclab.utils.math import quat_apply


def link_jacobian_to_point(
    link_jacobian_w: torch.Tensor,
    link_orientation_w_xyzw: torch.Tensor,
    offset_position_link: torch.Tensor,
) -> torch.Tensor:
    """Shift a world-frame spatial Jacobian from a link origin to a rigid point."""
    offset_position_w = quat_apply(link_orientation_w_xyzw, offset_position_link)
    angular = link_jacobian_w[..., 3:, :]
    linear = link_jacobian_w[..., :3, :] + torch.linalg.cross(
        angular.transpose(-1, -2),
        offset_position_w.unsqueeze(-2),
        dim=-1,
    ).transpose(-1, -2)
    return torch.cat((linear, angular), dim=-2)
