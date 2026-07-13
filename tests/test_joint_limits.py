"""Anti-windup IK target clamping (envs/door_task/joint_limits.py, no Kit)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from alexdoor_xas.envs.door_task.joint_limits import clamp_joint_targets  # noqa: E402


def test_clamp_is_bitwise_noop_inside_limits():
    lower = torch.tensor([[-1.0, -2.0, -0.5]])
    upper = torch.tensor([[1.0, 2.0, 0.5]])
    targets = torch.tensor([[0.3, -1.9, 0.5]])  # includes exactly-at-limit
    clamped, excess = clamp_joint_targets(targets, lower, upper)
    assert torch.equal(clamped, targets)
    assert torch.equal(excess, torch.zeros_like(targets))


def test_clamp_pins_exactly_to_limit_with_correct_excess():
    lower = torch.tensor([[-1.0, -2.0]])
    upper = torch.tensor([[1.0, 2.0]])
    targets = torch.tensor([[1.41, -2.25]])  # above upper / below lower
    clamped, excess = clamp_joint_targets(targets, lower, upper)
    assert torch.equal(clamped, torch.tensor([[1.0, -2.0]]))
    assert torch.allclose(excess, torch.tensor([[0.41, 0.25]]))
    # excess > 0 is the clamp-event flag
    assert bool((excess > 0).all())


def test_clamp_batched_mixed_envs():
    lower = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    upper = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0, 1.5], [-1.2, 0.9]])
    clamped, excess = clamp_joint_targets(targets, lower, upper)
    assert torch.equal(clamped, torch.tensor([[0.0, 1.0], [-1.0, 0.9]]))
    assert torch.allclose(excess, torch.tensor([[0.0, 0.5], [0.2, 0.0]]))


def test_clamp_rejects_inverted_limits():
    with pytest.raises(ValueError, match="lower bounds"):
        clamp_joint_targets(
            torch.zeros((1, 2)),
            torch.tensor([[0.5, -1.0]]),
            torch.tensor([[-0.5, 1.0]]),
        )
