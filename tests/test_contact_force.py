"""Raw-contact and terminal-force contracts."""

from __future__ import annotations

import math

import pytest
import torch

from alexdoor_xas.data_engine import plan_episodes, run_episode
from alexdoor_xas.envs.door_task.contact_force import sum_actor_contact_force
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from conftest import FakeDoorPushEnv, make_test_engine_cfg


def _inputs():
    return (
        torch.tensor([2.0, 3.0, 5.0, 7.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        torch.tensor([4]),
        torch.tensor([0]),
        torch.tensor([10, 99, 20, 99]),
        10,
    )


def test_sum_actor_contact_force_selects_exact_actor() -> None:
    result = sum_actor_contact_force(*_inputs())
    assert torch.equal(result, torch.tensor([[2.0, 0.0, 0.0]]))


def test_sum_actor_contact_force_returns_zero_until_actor_is_known() -> None:
    inputs = (*_inputs()[:-1], None)
    assert torch.equal(sum_actor_contact_force(*inputs), torch.zeros((1, 3)))


def test_sum_actor_contact_force_preserves_signed_physx_scalar() -> None:
    inputs = list(_inputs())
    inputs[0][0] = -2.0
    assert torch.equal(sum_actor_contact_force(*inputs), torch.tensor([[-2.0, 0.0, 0.0]]))


def test_sum_actor_contact_force_rejects_invalid_active_data() -> None:
    inputs = list(_inputs())
    inputs[0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sum_actor_contact_force(*inputs)

    inputs = list(_inputs())
    inputs[2] = torch.tensor([5])
    with pytest.raises(ValueError, match="exceeds buffer"):
        sum_actor_contact_force(*inputs)


def _episode():
    return run_episode(FakeDoorPushEnv(), plan_episodes(1, 0, 0)[0], make_test_engine_cfg())


def test_terminal_contact_recorded_with_alignment() -> None:
    episode = _episode()
    terminal = episode.extras["terminal_contact"]
    assert math.isfinite(terminal["force_n"]) and terminal["force_n"] >= 0.0
    assert isinstance(terminal["sensed"], bool)
    assert terminal["t"] == pytest.approx(episode.n_steps * episode.meta.control_dt)
    assert "response to" in terminal["alignment"]
    assert "pre-action" in terminal["alignment"]


def test_terminal_force_spike_fails_admission() -> None:
    episode = _episode()
    episode.extras["terminal_contact"]["force_n"] = FORCE_DATASET_LIMIT_N + 50.0
    diagnostics = contact_force_diagnostics(episode, force_limit_n=FORCE_DATASET_LIMIT_N)
    assert diagnostics["terminal"]["within_limit"] is False
    assert diagnostics["force_admission_passed"] is False
    result = check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N)
    assert any("terminal contact force" in error for error in result.errors)
    assert diagnostics["ticks_over_limit"] == []


def test_terminal_force_within_bound_passes() -> None:
    episode = _episode()
    diagnostics = contact_force_diagnostics(episode, force_limit_n=FORCE_DATASET_LIMIT_N)
    assert diagnostics["terminal"]["passed"] is True
    assert diagnostics["force_admission_passed"] is True
    assert check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N).ok


def test_non_finite_terminal_force_is_an_error() -> None:
    episode = _episode()
    episode.extras["terminal_contact"]["force_n"] = float("nan")
    diagnostics = contact_force_diagnostics(episode, force_limit_n=FORCE_DATASET_LIMIT_N)
    assert diagnostics["terminal"]["finite"] is False
    assert diagnostics["force_admission_passed"] is False
    assert any("non-finite" in error for error in check_alex_episode(episode).errors)


def test_legacy_episode_without_terminal_sample_stays_readable() -> None:
    episode = _episode()
    del episode.extras["terminal_contact"]
    diagnostics = contact_force_diagnostics(episode, force_limit_n=FORCE_DATASET_LIMIT_N)
    assert diagnostics["terminal"] is None
    assert diagnostics["force_admission_passed"] is True


def test_terminal_over_warn_without_error_limit_warns() -> None:
    episode = _episode()
    episode.extras["terminal_contact"]["force_n"] = FORCE_DATASET_LIMIT_N + 1.0
    result = check_alex_episode(episode)
    assert result.ok
    assert any("terminal contact force spiked" in warning for warning in result.warnings)
