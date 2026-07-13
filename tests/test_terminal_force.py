"""Terminal post-action force admission (post-3.3 review WP7)."""

from __future__ import annotations

import importlib.util
import math

import pytest

from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from conftest import FakeForceDoorPushEnv

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)


def _episode():
    return run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())


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


@requires_h5py
def test_terminal_contact_round_trips_through_hdf5(tmp_path) -> None:
    from alexdoor_xas.recording import read_episode, write_episode

    episode = _episode()
    path = write_episode(episode, tmp_path)
    loaded = read_episode(path)
    assert loaded.extras["terminal_contact"]["force_n"] == pytest.approx(
        episode.extras["terminal_contact"]["force_n"]
    )
    assert loaded.extras["terminal_contact"]["alignment"] == (
        episode.extras["terminal_contact"]["alignment"]
    )
