"""Pure metrics, report, sanity, and checkpoint-evaluation metadata contracts."""

from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest

from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.eval import FAILURE_LABELS, aggregate_metrics, episode_metrics, label_episode
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.common.eval_metadata import checkpoint_metadata
from alexdoor_xas.policies.scripted import DoorPushControllerCfg
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv

# --- test_eval ---


def _episode(controller_cfg: DoorPushControllerCfg | None = None, seed: int = 0):
    env = FakeDoorPushEnv(controller_cfg=controller_cfg)
    return run_episode(
        env, plan_episodes(1, 0, seed)[0], DataEngineCfg(), controller_cfg=controller_cfg
    )


FORCE_METRIC_KEYS = (
    "mean_contact_force_n",
    "max_contact_force_n",
    "p95_contact_force_n",
    "max_force_t_s",
    "max_force_phase",
    "first_contact_t_s",
    "contact_force_impulse_ns",
)


def test_metrics_prefer_force_sensed_contact() -> None:
    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    metrics = episode_metrics(episode)

    sensed_ticks = sum(1 for step in episode.steps if step.contact["sensed"])
    assert metrics["contact_ticks"] == sensed_ticks
    assert metrics["mean_contact_force_n"] is not None
    assert metrics["mean_contact_force_n"] > 0.0

    # Episodes without force sensing keep the geometric count and report no force.
    legacy = _episode()
    legacy_metrics = episode_metrics(legacy)
    inferred_ticks = sum(1 for step in legacy.steps if step.contact["inferred"])
    assert legacy_metrics["contact_ticks"] == inferred_ticks
    assert all(legacy_metrics[key] is None for key in FORCE_METRIC_KEYS)


def test_force_metrics_details_and_aggregate_block() -> None:
    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    m = episode_metrics(episode)

    sensed = [float(s.contact["force_n"]) for s in episode.steps if s.contact["sensed"]]
    assert m["max_contact_force_n"] == pytest.approx(max(sensed))
    assert m["mean_contact_force_n"] <= m["p95_contact_force_n"] <= m["max_contact_force_n"]
    assert 0.0 <= m["first_contact_t_s"] <= m["max_force_t_s"]
    assert m["max_force_phase"] in m["phase_ticks"]
    assert m["contact_force_impulse_ns"] == pytest.approx(sum(sensed) * episode.meta.control_dt)

    # The force peaks while the arm is actually driving the door.
    assert m["max_force_phase"] in ("contact", "push", "hold")

    summary = aggregate_metrics([m, m])
    block = summary["contact_force_n"]
    assert block["mean_of_means"] == pytest.approx(m["mean_contact_force_n"])
    assert block["max"] == pytest.approx(m["max_contact_force_n"])
    assert block["p95_max"] == pytest.approx(m["p95_contact_force_n"])
    assert block["mean_contact_ticks"] == pytest.approx(m["contact_ticks"])

    # Proxy-only runs get no force block at all.
    proxy_summary = aggregate_metrics([episode_metrics(_episode())])
    assert "contact_force_n" not in proxy_summary


def test_label_episode_success_is_none() -> None:
    assert (
        label_episode(
            final_angle_rad=1.0,
            success_angle_rad=math.pi / 4,
            controller_done=True,
            timed_out=False,
            last_phase="done",
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                final_angle_rad=float("nan"),
                controller_done=False,
                timed_out=False,
                last_phase="push",
            ),
            "non_finite_state",
        ),
        (
            dict(final_angle_rad=0.1, controller_done=False, timed_out=True, last_phase="contact"),
            "phase_timeout_contact",
        ),
        (
            dict(final_angle_rad=0.1, controller_done=False, timed_out=False, last_phase="push"),
            "env_truncated_before_completion",
        ),
        (
            dict(final_angle_rad=0.1, controller_done=True, timed_out=False, last_phase="done"),
            "insufficient_final_angle",
        ),
        (
            dict(
                final_angle_rad=0.2,
                controller_done=False,
                timed_out=False,
                last_phase="push",
                notes="env.step failed: NaN",
            ),
            "non_finite_state",
        ),
    ],
)
def test_label_episode_failure_cases(kwargs, expected) -> None:
    label = label_episode(success_angle_rad=math.pi / 4, **kwargs)
    assert label == expected
    assert label in FAILURE_LABELS


def test_episode_and_aggregate_metrics_on_success_and_timeout() -> None:
    ok = _episode()
    # An unreachable contact budget forces phase_timeout_contact.
    timeout_cfg = DoorPushControllerCfg(contact_max_ticks=1, contact_eps_m=-1.0)
    failed = _episode(controller_cfg=timeout_cfg, seed=1)

    m_ok = episode_metrics(ok)
    assert m_ok["success"] is True
    assert m_ok["failure_label"] is None
    assert m_ok["time_to_threshold_s"] is not None
    assert m_ok["max_door_angle_rad"] >= math.pi / 4
    assert m_ok["contact_ticks"] > 0
    assert m_ok["phase_ticks"]["approach"] > 0

    m_failed = episode_metrics(failed)
    assert m_failed["success"] is False
    assert m_failed["failure_label"] == "phase_timeout_contact"

    summary = aggregate_metrics([m_ok, m_failed])
    assert summary["n_episodes"] == 2
    assert summary["n_success"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["failure_labels"] == {"phase_timeout_contact": 1}
    assert summary["final_door_angle_rad"]["max"] >= math.pi / 4


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"
)
def test_plots_are_written(tmp_path) -> None:
    from alexdoor_xas.eval.plots import door_angle_plot, final_angle_plot

    episodes = [_episode(), _episode(seed=5)]
    angle_path = door_angle_plot(episodes, tmp_path / "plots" / "door_angle.png")
    final_path = final_angle_plot(episodes, tmp_path / "plots" / "final_angle.png")
    assert angle_path.is_file() and angle_path.stat().st_size > 0
    assert final_path.is_file() and final_path.stat().st_size > 0


def test_run_report_is_written(tmp_path) -> None:
    from alexdoor_xas.eval.report import write_run_report

    episodes = [_episode()]
    metrics = [episode_metrics(episodes[0])]
    report = write_run_report(
        tmp_path / "report.md",
        episodes=episodes,
        per_episode_metrics=metrics,
        aggregate=aggregate_metrics(metrics),
        exports=None,
        plots=None,
        videos={"status": "rendering unavailable in this shell", "files": []},
        limitations=["A1 export placeholder"],
    )
    text = report.read_text()
    assert "Scripted door-push baseline" in text
    # Proxy episodes: A1 unavailable because the proxy has no joints, no force columns.
    assert "`A1_joint_delta`: **not exported** — the proxy end-effector has no joints" in text
    assert "max force (N)" not in text
    assert "rendering unavailable in this shell" in text


def _write_report(tmp_path, episodes, exports):
    from alexdoor_xas.eval.report import write_run_report

    metrics = [episode_metrics(episode) for episode in episodes]
    report = write_run_report(
        tmp_path / "report.md",
        episodes=episodes,
        per_episode_metrics=metrics,
        aggregate=aggregate_metrics(metrics),
        exports=exports,
        plots=None,
        videos={"status": "not requested", "files": []},
        limitations=[],
    )
    return report.read_text()


def test_run_report_a1_line_and_force_columns_for_alex_episodes(tmp_path) -> None:
    from pathlib import Path

    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())

    # Joint-recording episodes with A1 exported: A1 listed like any other space.
    exports = {
        "A2_ee_delta": Path("/tmp/a2"),
        "A1_joint_delta": Path("/tmp/a1"),
    }
    text = _write_report(tmp_path / "with_a1", [episode], exports)
    assert "- `A1_joint_delta` → `/tmp/a1`" in text
    assert "**not exported" not in text
    # Force evidence is surfaced in the episode table.
    assert "contact ticks | mean force (N) | max force (N)" in text

    # Joint-recording episodes without an A1 export: relabelable, not proxy wording.
    text = _write_report(tmp_path / "without_a1", [episode], {"A2_ee_delta": Path("/tmp/a2")})
    assert "**not exported in this run**" in text
    assert "so A1 is relabelable" in text
    assert "the proxy end-effector has no joints" not in text


def test_sanity_checks_pass_on_clean_episode_and_catch_bad_data() -> None:
    import numpy as np

    from alexdoor_xas.eval import check_alex_episode, contact_force_diagnostics

    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    result = check_alex_episode(episode)
    assert result.ok and not result.warnings

    # Non-finite joint state.
    import dataclasses

    def with_step(base, index, **changes):
        clone = type(base)(meta=base.meta, steps=list(base.steps), extras=dict(base.extras))
        clone.steps[index] = dataclasses.replace(base.steps[index], **changes)
        clone.outcome = base.outcome
        return clone

    bad_proprio = dict(episode.steps[3].proprio)
    bad_proprio["joint_pos"] = np.full_like(bad_proprio["joint_pos"], np.nan)
    result = check_alex_episode(with_step(episode, 3, proprio=bad_proprio))
    assert any("non-finite joint_pos" in error for error in result.errors)

    # Joint target far outside the recorded position limits: error.
    bad_proprio = dict(episode.steps[3].proprio)
    bad_target = bad_proprio["joint_pos_target"].copy()
    bad_target[0] = 99.0
    bad_proprio["joint_pos_target"] = bad_target
    result = check_alex_episode(with_step(episode, 3, proprio=bad_proprio))
    assert any("exceeds its position limits" in error for error in result.errors)

    # Small overshoot (unclamped diff-IK drift at the limit): warning only.
    drift_proprio = dict(episode.steps[3].proprio)
    drift_target = drift_proprio["joint_pos_target"].copy()
    drift_target[0] = 2.5 + 0.05  # fake limits are [-2.5, 2.5]; band is (0.01, 0.1]
    drift_proprio["joint_pos_target"] = drift_target
    result = check_alex_episode(with_step(episode, 3, proprio=drift_proprio))
    assert result.ok
    assert any("diff-IK drift" in warning for warning in result.warnings)

    # Joint velocity above the recorded sim limit (after the settle window).
    bad_proprio = dict(episode.steps[40].proprio)
    bad_vel = bad_proprio["joint_vel"].copy()
    bad_vel[2] = 50.0
    bad_proprio["joint_vel"] = bad_vel
    result = check_alex_episode(with_step(episode, 40, proprio=bad_proprio))
    assert any("above its sim limit" in error for error in result.errors)

    # The same spike inside the settle window is the reset transient: ignored.
    result = check_alex_episode(with_step(episode, 3, proprio=bad_proprio))
    assert result.ok

    # Arm velocity below the sim limit but above the task cap is only a warning.
    warm_proprio = dict(episode.steps[40].proprio)
    warm_vel = warm_proprio["joint_vel"].copy()
    warm_vel[1] = 5.0  # sim limit 10.0, task cap 4.0
    warm_proprio["joint_vel"] = warm_vel
    result = check_alex_episode(with_step(episode, 40, proprio=warm_proprio))
    assert result.ok
    assert any("rad/s" in warning for warning in result.warnings)

    # Wrong contact source (e.g. a proxy episode routed into the Alex gate).
    bad_contact = dict(episode.steps[0].contact)
    bad_contact["source"] = "inferred_geometric"
    result = check_alex_episode(with_step(episode, 0, contact=bad_contact))
    assert any("contact source" in error for error in result.errors)

    # Force spike warning.
    spike_contact = dict(episode.steps[3].contact)
    spike_contact["force_n"] = 500.0
    result = check_alex_episode(with_step(episode, 3, contact=spike_contact))
    assert result.ok
    assert any("spiked to 500.0 N" in warning for warning in result.warnings)

    evidence = contact_force_diagnostics(
        with_step(episode, 3, contact=spike_contact), force_limit_n=200.0
    )
    assert evidence["max_force_tick"] == 3
    assert evidence["ticks_over_limit"] == [3]
    assert evidence["one_tick_over_limit"] is True
    assert evidence["sustained_over_limit"] is False
    assert evidence["peak"]["causal_action_tick"] == 2
    assert evidence["peak"]["causal_action_phase"] == episode.steps[2].safety["controller_phase"]

    # The same threshold is a hard admission limit when a dataset/gate caller opts in.
    result = check_alex_episode(with_step(episode, 3, contact=spike_contact), force_error_n=200.0)
    assert not result.ok
    assert any("exceeded the 200 N force admission limit" in error for error in result.errors)

    # Proxy episodes (no joint proprio) are rejected outright.
    result = check_alex_episode(_episode())
    assert not result.ok


def test_sanity_checker_rejects_negative_force_magnitude_with_diagnostics() -> None:
    import dataclasses

    from alexdoor_xas.eval import check_alex_episode, contact_force_diagnostics

    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = -2.5
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)

    result = check_alex_episode(episode)
    assert not result.ok
    assert any("force magnitude must be non-negative" in error for error in result.errors)

    evidence = contact_force_diagnostics(episode)
    assert evidence["min_force_n"] == -2.5
    assert evidence["min_force_tick"] == 3
    assert evidence["negative_force_ticks"] == [3]
    assert evidence["non_negative_force_gate_passed"] is False
    assert evidence["force_admission_passed"] is False


@pytest.mark.parametrize(
    "force_n",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive_inf", "negative_inf"),
)
def test_sanity_checker_rejects_non_finite_force_magnitude(force_n: float) -> None:
    import dataclasses

    from alexdoor_xas.eval import check_alex_episode, contact_force_diagnostics

    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = force_n
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)

    result = check_alex_episode(episode, force_error_n=200.0)
    assert not result.ok
    assert any("non-finite contact force values" in error for error in result.errors)

    evidence = contact_force_diagnostics(episode)
    assert evidence["all_forces_finite"] is False
    assert evidence["non_finite_force_ticks"] == [3]
    assert evidence["force_admission_passed"] is False


def test_force_admission_accepts_exact_limit_and_preserves_lower_warning() -> None:
    import dataclasses

    from alexdoor_xas.eval import check_alex_episode, contact_force_diagnostics

    episode = run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, 0)[0], DataEngineCfg())
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = 200.0
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)

    diagnostic = check_alex_episode(episode, force_warn_n=150.0)
    assert diagnostic.ok
    assert any("warn threshold 150 N" in warning for warning in diagnostic.warnings)

    admission = check_alex_episode(episode, force_error_n=200.0)
    assert admission.ok
    evidence = contact_force_diagnostics(episode, force_limit_n=200.0)
    assert evidence["max_force_n"] == 200.0
    assert evidence["force_admission_passed"] is True
    assert evidence["ticks_over_limit"] == []


# --- test_eval_metadata ---


def test_checkpoint_metadata_is_self_contained_and_dataset_independent() -> None:
    policy = SimpleNamespace(
        checkpoint_format="alexdoor_xas.act.v2",
        checkpoint_config={
            "dataset": {
                "task": "door_push_alex_v2",
                "space": "A2_ee_delta",
                "version": "v3_scale_master",
                "view_id": "v3_scale_n50",
                "obs_preset": "core_door_pose",
            }
        },
        action_space="A2_ee_delta",
        obs_preset="core_door_pose",
        model=SimpleNamespace(
            obs_dim=16,
            action_dim=6,
            cfg=ActModelCfg(chunk_size=8),
        ),
    )

    result = checkpoint_metadata(policy, "act")

    assert result["format"] == "alexdoor_xas.act.v2"
    assert result["policy"] == "act"
    assert result["dataset"]["view_id"] == "v3_scale_n50"
    assert result["observation_dim"] == 16
    assert result["action_dim"] == 6
    assert result["model_config"]["chunk_size"] == 8
    assert "dataset_provenance" not in result
