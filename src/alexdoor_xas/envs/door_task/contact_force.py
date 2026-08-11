"""GPU tensor helpers for exact contact-partner force aggregation."""

from __future__ import annotations

import torch


def sum_actor_contact_forces(
    force_magnitudes: torch.Tensor,
    normals_w: torch.Tensor,
    counts: torch.Tensor,
    start_indices: torch.Tensor,
    other_actor_ids: torch.Tensor,
    target_actor_ids: torch.Tensor,
    target_known: torch.Tensor,
) -> torch.Tensor:
    """Sum raw contact forces matching one target actor per sensor.

    All inputs and the result stay on the same device. Unused raw-buffer slots
    are excluded through ``counts`` and ``start_indices`` before validation.
    """

    force_magnitudes = force_magnitudes.reshape(-1)
    other_actor_ids = other_actor_ids.reshape(-1)
    counts = counts.reshape(-1).to(dtype=torch.long)
    start_indices = start_indices.reshape(-1).to(dtype=torch.long)
    target_actor_ids = target_actor_ids.reshape(-1).to(dtype=torch.long)
    target_known = target_known.reshape(-1).to(dtype=torch.bool)

    capacity = force_magnitudes.numel()
    num_sensors = counts.numel()
    if normals_w.shape != (capacity, 3):
        raise ValueError(
            f"raw contact normals must have shape ({capacity}, 3), got {tuple(normals_w.shape)}"
        )
    if other_actor_ids.numel() != capacity:
        raise ValueError("raw contact actor IDs must match force-buffer capacity")
    expected_sensor_shape = (num_sensors,)
    if start_indices.shape != expected_sensor_shape:
        raise ValueError("raw contact start indices must match sensor count")
    if target_actor_ids.shape != expected_sensor_shape or target_known.shape != (
        num_sensors,
    ):
        raise ValueError("target actor IDs and validity flags must match sensor count")
    if bool((counts < 0).any()) or bool((start_indices < 0).any()):
        raise ValueError("raw contact ranges must be non-negative")
    if bool((start_indices + counts > capacity).any()):
        raise ValueError("raw contact range exceeds buffer capacity")

    slot = torch.arange(capacity, device=force_magnitudes.device)
    active = (slot.unsqueeze(0) >= start_indices.unsqueeze(1)) & (
        slot.unsqueeze(0) < (start_indices + counts).unsqueeze(1)
    )
    if bool(active.any()):
        active_slots = active.any(dim=0)
        if not bool(torch.isfinite(force_magnitudes[active_slots]).all()):
            raise ValueError("raw contact force contains non-finite values")
        if bool((force_magnitudes[active_slots] < 0).any()):
            raise ValueError("raw contact force magnitude must be non-negative")
        if not bool(torch.isfinite(normals_w[active_slots]).all()):
            raise ValueError("raw contact normal contains non-finite values")

    matches = (
        active
        & target_known.unsqueeze(1)
        & (other_actor_ids.to(dtype=torch.long).unsqueeze(0) == target_actor_ids.unsqueeze(1))
    )
    vectors_w = force_magnitudes.unsqueeze(1) * normals_w
    return (matches.unsqueeze(2) * vectors_w.unsqueeze(0)).sum(dim=1)


__all__ = ["sum_actor_contact_forces"]
