"""Raw-contact, terminal-force, and force-admission contracts."""

from __future__ import annotations

import math

import pytest
import torch

from alexdoor_xas.data_engine import plan_episodes, run_episode
from alexdoor_xas.envs.door_task.contact_force import (
    decode_contact_flag,
    sum_actor_contact_forces,
)
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from conftest import FakeForceDoorPushEnv, make_test_engine_cfg

# --- test_contact_force ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, False),
        (True, True),
        (0, False),
        (1.0, True),
        (torch.tensor([0]), False),
        (torch.tensor([1.0]), True),
    ],
)
def test_decode_contact_flag_accepts_only_exact_binary_scalars(value, expected) -> None:
    assert decode_contact_flag(value) is expected


@pytest.mark.parametrize(
    "value",
    [-1, 0.5, 2, float("nan"), float("inf"), [], [0, 1]],
)
def test_decode_contact_flag_rejects_ambiguous_values(value) -> None:
    with pytest.raises(ValueError, match="contact flag"):
        decode_contact_flag(value)


def _inputs():
    forces = torch.tensor([2.0, 3.0, 5.0, 7.0])
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
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


def test_sum_actor_contact_forces_preserves_signed_physx_scalars() -> None:
    inputs = list(_inputs())
    inputs[0][0] = -2.0

    result = sum_actor_contact_forces(*inputs)

    assert torch.equal(result, torch.tensor([[-2.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))


def test_sum_actor_contact_forces_rejects_invalid_active_data() -> None:
    inputs = list(_inputs())
    inputs[0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sum_actor_contact_forces(*inputs)

    inputs = list(_inputs())
    inputs[2] = torch.tensor([5, 2])
    with pytest.raises(ValueError, match="exceeds buffer"):
        sum_actor_contact_forces(*inputs)


# --- test_terminal_force ---


def _episode():
    return run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], make_test_engine_cfg())


class _InvalidContactFlagEnv(FakeForceDoorPushEnv):
    def contact_sensed(self):
        return torch.tensor([0.5])


def test_run_episode_rejects_nonbinary_contact_flag() -> None:
    with pytest.raises(ValueError, match="exactly 0/1"):
        run_episode(
            _InvalidContactFlagEnv(),
            plan_episodes(1, 0, 0)[0],
            make_test_engine_cfg(),
        )


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
    # Per-step samples alone would have passed: the spike is terminal-only.
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
    result = check_alex_episode(episode)
    assert any("non-finite" in error for error in result.errors)


def test_episodes_without_terminal_sample_stay_readable() -> None:
    # phase2.v0/v1 episodes recorded before the terminal extra existed.
    episode = _episode()
    del episode.extras["terminal_contact"]
    diagnostics = contact_force_diagnostics(episode, force_limit_n=FORCE_DATASET_LIMIT_N)
    assert diagnostics["terminal"] is None
    assert diagnostics["force_admission_passed"] is True
    assert check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N).ok


def test_terminal_over_warn_without_error_limit_warns() -> None:
    episode = _episode()
    episode.extras["terminal_contact"]["force_n"] = FORCE_DATASET_LIMIT_N + 1.0
    result = check_alex_episode(episode)  # no force_error_n: diagnostic caller
    assert result.ok
    assert any("terminal contact force spiked" in warning for warning in result.warnings)
