"""Operational evaluation contracts."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.data_engine import plan_episodes, run_episode
from alexdoor_xas.eval.metrics import aggregate_metrics, episode_metrics
from alexdoor_xas.eval.sanity import check_alex_episode
from alexdoor_xas.policies.scripted import DoorPushControllerCfg
from conftest import FakeDoorPushEnv, make_test_engine_cfg

_EPISODE_METRIC_KEYS = {
    "episode_id",
    "seed",
    "randomized",
    "success",
    "termination_reason",
    "environment_terminated",
    "environment_truncated",
    "final_door_angle_rad",
    "max_door_angle_rad",
    "time_to_threshold_s",
    "n_steps",
    "duration_s",
    "contact_ticks",
    "mean_contact_force_n",
    "max_contact_force_n",
}


def _episode(controller_cfg: DoorPushControllerCfg | None = None, seed: int = 0):
    return run_episode(
        FakeDoorPushEnv(controller_cfg=controller_cfg),
        plan_episodes(1, 0, seed)[0],
        make_test_engine_cfg(),
        controller_cfg=controller_cfg,
    )


def _with_step(episode, index, **changes):
    clone = type(episode)(meta=episode.meta, steps=list(episode.steps), extras=dict(episode.extras))
    clone.steps[index] = dataclasses.replace(episode.steps[index], **changes)
    clone.outcome = episode.outcome
    return clone


def test_episode_and_aggregate_metrics_are_compact_and_factual() -> None:
    successful = _episode()
    timeout_cfg = DoorPushControllerCfg(contact_max_ticks=1, contact_eps_m=-1.0)
    failed = _episode(controller_cfg=timeout_cfg, seed=1)

    success_metrics = episode_metrics(successful)
    failed_metrics = episode_metrics(failed)
    assert set(success_metrics) == _EPISODE_METRIC_KEYS
    assert success_metrics["success"] is True
    assert success_metrics["termination_reason"] == "controller_done"
    assert success_metrics["time_to_threshold_s"] is not None
    assert success_metrics["max_door_angle_rad"] >= math.pi / 4
    assert success_metrics["contact_ticks"] > 0
    assert 0.0 < success_metrics["mean_contact_force_n"] <= success_metrics["max_contact_force_n"]
    assert failed_metrics["success"] is False
    assert failed_metrics["termination_reason"] == "controller_timeout"

    aggregate = aggregate_metrics([success_metrics, failed_metrics])
    assert aggregate["n_episodes"] == 2
    assert aggregate["n_success"] == 1
    assert aggregate["success_rate"] == pytest.approx(0.5)
    assert aggregate["termination_reasons"] == {
        "controller_done": 1,
        "controller_timeout": 1,
    }
    assert set(aggregate["contact_force_n"]) == {
        "mean_of_means",
        "max",
        "mean_contact_ticks",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"
)
def test_plots_and_report_are_written(tmp_path) -> None:
    from alexdoor_xas.eval.plots import door_angle_plot, final_angle_plot
    from alexdoor_xas.eval.report import write_run_report

    episodes = [_episode()]
    metrics = [episode_metrics(episodes[0])]
    plots = {
        "door_angle_vs_time": door_angle_plot(episodes, tmp_path / "plots/door_angle.png"),
        "final_door_angle": final_angle_plot(episodes, tmp_path / "plots/final_angle.png"),
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in plots.values())

    report = write_run_report(
        tmp_path / "report.md",
        episodes=episodes,
        per_episode_metrics=metrics,
        aggregate=aggregate_metrics(metrics),
        exports={"A2_ee_delta": Path("/tmp/a2")},
        plots=plots,
        videos={"status": "not requested", "files": []},
        limitations=["Fixed-base benchmark."],
    )
    text = report.read_text()
    assert "Scripted door-push baseline" in text
    assert "contact ticks | mean force (N) | max force (N)" in text
    assert "`A1_joint_delta`: **not exported in this run**" in text
    assert "recorded joint targets keep A1 relabelable" in text
    assert "Fixed-base benchmark." in text


def test_sanity_checks_joint_state_limits_velocity_and_contact_source() -> None:
    episode = _episode()
    assert check_alex_episode(episode).ok

    bad_proprio = dict(episode.steps[3].proprio)
    bad_proprio["joint_pos"] = np.full_like(bad_proprio["joint_pos"], np.nan)
    result = check_alex_episode(_with_step(episode, 3, proprio=bad_proprio))
    assert any("non-finite joint_pos" in error for error in result.errors)

    bad_proprio = dict(episode.steps[3].proprio)
    bad_target = bad_proprio["joint_pos_target"].copy()
    bad_target[0] = 99.0
    bad_proprio["joint_pos_target"] = bad_target
    result = check_alex_episode(_with_step(episode, 3, proprio=bad_proprio))
    assert any("exceeds its position limits" in error for error in result.errors)

    drift_proprio = dict(episode.steps[3].proprio)
    drift_target = drift_proprio["joint_pos_target"].copy()
    drift_target[0] = 2.55
    drift_proprio["joint_pos_target"] = drift_target
    result = check_alex_episode(_with_step(episode, 3, proprio=drift_proprio))
    assert result.ok and result.warnings

    bad_proprio = dict(episode.steps[40].proprio)
    bad_velocity = bad_proprio["joint_vel"].copy()
    bad_velocity[2] = 50.0
    bad_proprio["joint_vel"] = bad_velocity
    result = check_alex_episode(_with_step(episode, 40, proprio=bad_proprio))
    assert any("above its sim limit" in error for error in result.errors)
    assert check_alex_episode(_with_step(episode, 3, proprio=bad_proprio)).ok

    bad_contact = dict(episode.steps[0].contact)
    bad_contact["source"] = "inferred_geometric"
    result = check_alex_episode(_with_step(episode, 0, contact=bad_contact))
    assert any("invalid contact source" in error for error in result.errors)
