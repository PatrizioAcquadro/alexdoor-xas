"""Tests for GPU-compatible raw-contact actor filtering."""

from __future__ import annotations

import pytest
import torch

from alexdoor_xas.envs.door_task.contact_force import sum_actor_contact_forces


def _inputs():
    forces = torch.tensor([2.0, 3.0, 5.0, 7.0])
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    counts = torch.tensor([2, 2])
    starts = torch.tensor([0, 2])
    actor_ids = torch.tensor([10, 99, 20, 99])
    targets = torch.tensor([10, 20])
    known = torch.tensor([True, True])
    return forces, normals, counts, starts, actor_ids, targets, known


def test_sum_actor_contact_forces_selects_one_actor_per_sensor() -> None:
    result = sum_actor_contact_forces(*_inputs())

    assert torch.equal(result, torch.tensor([[2.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))


def test_sum_actor_contact_forces_ignores_unknown_target_and_other_contacts() -> None:
    inputs = list(_inputs())
    inputs[-1] = torch.tensor([False, True])

    result = sum_actor_contact_forces(*inputs)

    assert torch.equal(result, torch.tensor([[0.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))


def test_sum_actor_contact_forces_rejects_invalid_active_data() -> None:
    inputs = list(_inputs())
    inputs[0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sum_actor_contact_forces(*inputs)

    inputs = list(_inputs())
    inputs[0][0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        sum_actor_contact_forces(*inputs)

    inputs = list(_inputs())
    inputs[2] = torch.tensor([5, 2])
    with pytest.raises(ValueError, match="exceeds buffer"):
        sum_actor_contact_forces(*inputs)
