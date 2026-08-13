"""Raw-contact and force-admission contracts."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from alexdoor_xas.data_engine import plan_episodes, run_episode
from alexdoor_xas.envs.door_task.contact_force import sum_actor_contact_force
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_summary,
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


def _episode():
    return run_episode(FakeDoorPushEnv(), plan_episodes(1, 0, 0)[0], make_test_engine_cfg())


def _set_step_force(episode, force_n: float) -> None:
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = force_n
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)


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


def test_terminal_contact_is_recorded_and_summarized() -> None:
    episode = _episode()
    terminal = episode.extras["terminal_contact"]
    summary = contact_force_summary(episode)

    assert math.isfinite(terminal["force_n"]) and terminal["force_n"] >= 0.0
    assert isinstance(terminal["sensed"], bool)
    assert terminal["t"] == pytest.approx(episode.n_steps * episode.meta.control_dt)
    assert set(summary) == {
        "force_limit_n",
        "max_force_n",
        "max_force_tick",
        "non_finite_ticks",
        "negative_ticks",
        "over_limit_ticks",
        "terminal",
    }
    assert summary["terminal"]["within_limit"] is True
    assert check_alex_episode(episode).ok


@pytest.mark.parametrize(
    ("force_n", "summary_key", "error_text"),
    [
        (float("nan"), "non_finite_ticks", "non-finite contact force"),
        (-2.5, "negative_ticks", "force magnitude is negative"),
        (FORCE_DATASET_LIMIT_N + 1.0, "over_limit_ticks", "admission limit"),
    ],
)
def test_step_force_gate_rejects_invalid_values(force_n, summary_key, error_text) -> None:
    episode = _episode()
    _set_step_force(episode, force_n)

    assert contact_force_summary(episode)[summary_key] == [3]
    assert any(error_text in error for error in check_alex_episode(episode).errors)


@pytest.mark.parametrize(
    ("force_n", "error_text"),
    [
        (float("nan"), "non-finite"),
        (-1.0, "magnitude is negative"),
        (FORCE_DATASET_LIMIT_N + 1.0, "admission limit"),
    ],
)
def test_terminal_force_gate_rejects_invalid_values(force_n, error_text) -> None:
    episode = _episode()
    episode.extras["terminal_contact"]["force_n"] = force_n

    assert contact_force_summary(episode)["terminal"]["within_limit"] is False
    assert any(error_text in error for error in check_alex_episode(episode).errors)


def test_force_limit_is_inclusive() -> None:
    episode = _episode()
    _set_step_force(episode, FORCE_DATASET_LIMIT_N)

    assert contact_force_summary(episode)["over_limit_ticks"] == []
    assert check_alex_episode(episode).ok


def test_legacy_episode_without_terminal_contact_remains_valid() -> None:
    episode = _episode()
    del episode.extras["terminal_contact"]

    assert contact_force_summary(episode)["terminal"] is None
    assert check_alex_episode(episode).ok
