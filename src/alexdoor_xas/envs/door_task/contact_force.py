"""GPU aggregation of raw contacts with one exact partner actor."""

from __future__ import annotations

import torch


def sum_actor_contact_force(
    force_magnitudes: torch.Tensor,
    normals_w: torch.Tensor,
    counts: torch.Tensor,
    start_indices: torch.Tensor,
    other_actor_ids: torch.Tensor,
    target_actor_id: int | None,
) -> torch.Tensor:
    """Return the single sensor's world-frame force against ``target_actor_id``."""

    force_magnitudes = force_magnitudes.reshape(-1)
    other_actor_ids = other_actor_ids.reshape(-1).to(dtype=torch.long)
    counts = counts.reshape(-1).to(dtype=torch.long)
    start_indices = start_indices.reshape(-1).to(dtype=torch.long)

    capacity = force_magnitudes.numel()
    if normals_w.shape != (capacity, 3) or other_actor_ids.numel() != capacity:
        raise ValueError("raw contact buffers have inconsistent shapes")
    if counts.shape != (1,) or start_indices.shape != (1,):
        raise ValueError("raw contact ranges must describe one sensor")
    if bool((counts < 0).any()) or bool((start_indices < 0).any()):
        raise ValueError("raw contact ranges must be non-negative")
    if bool((start_indices + counts > capacity).any()):
        raise ValueError("raw contact range exceeds buffer capacity")

    slots = torch.arange(capacity, device=force_magnitudes.device)
    active = (slots >= start_indices[0]) & (slots < start_indices[0] + counts[0])
    if bool(active.any()):
        if not bool(torch.isfinite(force_magnitudes[active]).all()):
            raise ValueError("raw contact force contains non-finite values")
        if not bool(torch.isfinite(normals_w[active]).all()):
            raise ValueError("raw contact normal contains non-finite values")
    if target_actor_id is None:
        return normals_w.new_zeros((1, 3))

    matches = active & (other_actor_ids == target_actor_id)
    # PhysX reports a signed coefficient relative to each contact normal.
    vectors_w = force_magnitudes.unsqueeze(1) * normals_w
    return torch.where(matches.unsqueeze(1), vectors_w, 0.0).sum(dim=0, keepdim=True)
