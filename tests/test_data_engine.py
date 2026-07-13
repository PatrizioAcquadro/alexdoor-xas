"""Pure tests for the data engine: episode generation loop + dataset exports."""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest

from alexdoor_xas.action.frames import door_frame_from_body_pose, world_delta_to_frame
from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)
from alexdoor_xas.data_engine import (
    DataEngineCfg,
    export_datasets,
    plan_episodes,
    run_episode,
    traces_equal,
)
from alexdoor_xas.recording import read_episode
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)


def test_plan_episodes_fixed_then_randomized_and_deterministic() -> None:
    plan_a = plan_episodes(n_fixed=2, n_randomized=3, base_seed=100)
    plan_b = plan_episodes(n_fixed=2, n_randomized=3, base_seed=100)
    assert plan_a == plan_b
    assert len(plan_a) == 5
    assert [item.variation is None for item in plan_a] == [True, True, False, False, False]
    assert [item.seed for item in plan_a] == [100, 101, 102, 103, 104]


def _generate(yaw: float = 0.3, seed: int = 0, variation=None):
    env = FakeDoorPushEnv(yaw_rad=yaw, origin=(1.0, -2.0, 0.5))
    plan = plan_episodes(1, 0, seed)[0]
    if variation is not None:
        plan = type(plan)(seed=seed, variation=variation)
    return run_episode(env, plan, DataEngineCfg())


def test_run_episode_succeeds_and_matches_schema() -> None:
    episode = _generate()

    assert episode.outcome is not None
    assert episode.outcome.success
    assert episode.outcome.failure_label is None
    assert episode.outcome.final_door_angle >= math.pi / 4
    assert episode.outcome.n_steps == episode.n_steps > 0
    assert episode.meta.action_space == A2_EE_DELTA
    assert episode.meta.robot == "proxy_ee_sphere_v0"
    assert episode.meta.control_dt == pytest.approx(1 / 60)

    assert episode.extras["action_door_frame"].shape == (episode.n_steps, 6)
    chunk_phases = [chunk["phase"] for chunk in episode.extras["a4_chunks"]]
    assert chunk_phases == [
        "approach", "align", "pre_contact", "contact", "push", "hold", "release",
    ]
    step = episode.steps[0]
    assert set(step.obs_ref) >= {"door_angle_rad", "ee_pos_x_m"}
    assert set(step.proprio) == {"ee_pos_w", "ee_quat_w_xyzw"}
    assert set(step.object_state) == {
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        "door_yaw_rad",
        "door_rel_pos_x",
        "door_rel_pos_y",
        "door_rel_pos_z",
    }
    assert step.contact["source"] == "inferred_geometric"
    assert step.safety["controller_phase"] == "approach"


def test_run_episode_is_deterministic() -> None:
    first = _generate(seed=3)
    second = _generate(seed=3)
    assert traces_equal(first, second, tol=0.0) == 0.0


def test_randomized_episode_records_variation_and_succeeds() -> None:
    variation = plan_episodes(0, 1, 42)[0].variation
    episode = _generate(seed=42, variation=variation)
    assert episode.extras["variation"] == variation.to_dict()
    assert episode.outcome.success
    assert episode.extras["controller_cfg"]["push_radius_frac"] == pytest.approx(
        variation.push_radius_frac
    )


@requires_h5py
def test_export_datasets_produces_a2_a3_a4(tmp_path) -> None:
    episode = _generate()
    exported = export_datasets([episode], tmp_path, version="v0")

    assert set(exported) == {A2_EE_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK}
    for directory in exported.values():
        meta = json.loads((directory / "meta.json").read_text())
        assert meta["task"] == "door_push"
        assert meta["n_episodes"] == 1
        assert meta["n_success"] == 1
        assert meta["generator"]["controller_cfg"] is not None

    a2_files = sorted(exported[A2_EE_DELTA].glob("episode_*.hdf5"))
    a3_files = sorted(exported[A3_OBJ_REL_EE_DELTA].glob("episode_*.hdf5"))
    assert len(a2_files) == len(a3_files) == 1

    a2 = read_episode(a2_files[0])
    a3 = read_episode(a3_files[0])
    assert a2.meta.action_space == A2_EE_DELTA
    assert a3.meta.action_space == A3_OBJ_REL_EE_DELTA

    # A3 actions must equal the A2 actions re-expressed in the recorded door frame.
    frame = door_frame_from_body_pose(
        np.asarray(a2.extras["door_frame_pos_w"]),
        np.asarray(a2.extras["door_frame_quat_w_xyzw"]),
    )
    for step_a2, step_a3 in zip(a2.steps, a3.steps, strict=True):
        np.testing.assert_allclose(
            step_a3.action, world_delta_to_frame(step_a2.action, frame), atol=1e-9
        )

    a4_lines = (exported[A4_OBJ_CENTRIC_CHUNK] / "episodes.jsonl").read_text().splitlines()
    assert len(a4_lines) == 1
    record = json.loads(a4_lines[0])
    assert record["meta"]["action_space"] == A4_OBJ_CENTRIC_CHUNK
    assert [chunk["phase"] for chunk in record["chunks"]][:2] == ["approach", "align"]
    assert record["outcome"]["success"] is True


def test_force_sensing_env_records_sensed_contact_and_joint_state() -> None:
    """The engine picks up the Phase 2.5 accessors via hasattr and records the
    force-sensed contact fields, joint proprio, and joint-name extras."""
    env = FakeForceDoorPushEnv()
    episode = run_episode(env, plan_episodes(1, 0, 0)[0], DataEngineCfg())

    assert episode.outcome is not None and episode.outcome.success
    step = episode.steps[0]
    assert step.contact["source"] == "force_sensor+geometric"
    assert set(step.contact) == {"inferred", "sensed", "force_n", "source"}
    assert set(step.proprio) == {
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "joint_pos",
        "joint_vel",
        "joint_pos_target",
    }
    assert step.proprio["joint_pos"].shape == (FakeForceDoorPushEnv.N_JOINTS,)
    sensed_ticks = [s for s in episode.steps if s.contact["sensed"]]
    assert sensed_ticks, "expected at least one force-sensed contact tick"
    assert all(s.contact["force_n"] > 0.0 for s in sensed_ticks)
    assert episode.extras["joint_names"] == env.robot_joint_names()
    assert episode.extras["arm_joint_ids"] == env.arm_joint_ids()
    # Hardening-pass extras: A1 diff closure + Isaac-reported joint limits.
    assert episode.extras["final_joint_pos_target"].shape == (FakeForceDoorPushEnv.N_JOINTS,)
    assert episode.extras["joint_pos_limits"].shape == (FakeForceDoorPushEnv.N_JOINTS, 2)
    assert episode.extras["joint_vel_limits"].shape == (FakeForceDoorPushEnv.N_JOINTS,)


def _force_episode(seed: int = 0):
    return run_episode(FakeForceDoorPushEnv(), plan_episodes(1, 0, seed)[0], DataEngineCfg())


@requires_h5py
def test_a1_export_relabels_joint_target_deltas(tmp_path) -> None:
    """Alex-style episodes additionally export A1 = joint-position-target deltas."""
    episode = _force_episode()
    exported = export_datasets([episode], tmp_path, version="v0")
    assert set(exported) == {
        A1_JOINT_DELTA,
        A2_EE_DELTA,
        A3_OBJ_REL_EE_DELTA,
        A4_OBJ_CENTRIC_CHUNK,
    }

    a1_files = sorted(exported[A1_JOINT_DELTA].glob("episode_*.hdf5"))
    assert len(a1_files) == 1
    a1 = read_episode(a1_files[0])
    assert a1.meta.action_space == A1_JOINT_DELTA
    actions = np.stack([step.action for step in a1.steps])
    assert actions.shape == (episode.n_steps, FakeForceDoorPushEnv.N_JOINTS)

    # Recorded targets are pre-step, so action[t] = target[t+1] - target[t] with
    # the last diff closed by extras["final_joint_pos_target"]. The fake env's
    # arm targets advance by TARGET_STEP_RAD per executed tick.
    arm = list(FakeForceDoorPushEnv().arm_joint_ids())
    held = [j for j in range(FakeForceDoorPushEnv.N_JOINTS) if j not in arm]
    np.testing.assert_allclose(
        actions[:, arm], FakeForceDoorPushEnv.TARGET_STEP_RAD, atol=1e-12
    )
    np.testing.assert_allclose(actions[:, held], 0.0, atol=1e-12)

    targets = np.stack([step.proprio["joint_pos_target"] for step in a1.steps])
    expected = np.diff(
        np.concatenate(
            [targets, episode.extras["final_joint_pos_target"].reshape(1, -1)], axis=0
        ),
        axis=0,
    )
    np.testing.assert_allclose(actions, expected, atol=1e-12)

    meta = json.loads((exported[A1_JOINT_DELTA] / "meta.json").read_text())
    assert meta["action_space"] == A1_JOINT_DELTA


@requires_h5py
def test_a1_export_without_final_target_falls_back_to_zero_last_delta(tmp_path) -> None:
    episode = _force_episode()
    del episode.extras["final_joint_pos_target"]  # pre-hardening episodes lack it
    exported = export_datasets([episode], tmp_path, version="v0")
    a1 = read_episode(sorted(exported[A1_JOINT_DELTA].glob("episode_*.hdf5"))[0])
    np.testing.assert_allclose(a1.steps[-1].action, 0.0, atol=1e-12)


def test_traces_equal_covers_joint_and_contact_traces() -> None:
    import dataclasses

    first = _force_episode()
    second = _force_episode()
    assert traces_equal(first, second, tol=0.0) == 0.0

    def with_mutated_step(episode, index, **changes):
        mutated = dataclasses.replace(episode.steps[index], **changes)
        clone = type(episode)(meta=episode.meta, steps=list(episode.steps),
                              extras=dict(episode.extras))
        clone.steps[index] = mutated
        clone.outcome = episode.outcome
        return clone

    # Joint-target divergence is caught even when action/EE/door traces match.
    bad_proprio = dict(second.steps[5].proprio)
    bad_proprio["joint_pos_target"] = bad_proprio["joint_pos_target"] + 1e-3
    with pytest.raises(AssertionError, match="joint_pos_target"):
        traces_equal(first, with_mutated_step(second, 5, proprio=bad_proprio), tol=1e-6)

    # Sensed-contact flags and controller phases are compared exactly.
    flipped = dict(second.steps[5].contact)
    flipped["sensed"] = not flipped["sensed"]
    with pytest.raises(AssertionError, match="contact.sensed"):
        traces_equal(first, with_mutated_step(second, 5, contact=flipped), tol=1e-6)
    bad_safety = dict(second.steps[5].safety)
    bad_safety["controller_phase"] = "warp"
    with pytest.raises(AssertionError, match="controller_phase"):
        traces_equal(first, with_mutated_step(second, 5, safety=bad_safety), tol=1e-6)

    # The contact force gets its own tolerance.
    noisy = dict(second.steps[5].contact)
    noisy["force_n"] = float(noisy["force_n"]) + 0.5
    noisy_episode = with_mutated_step(second, 5, contact=noisy)
    with pytest.raises(AssertionError, match="force_n"):
        traces_equal(first, noisy_episode, tol=1e-6)
    assert traces_equal(first, noisy_episode, tol=1e-6, force_tol=1.0) <= 1.0


@requires_h5py
def test_run_baseline_same_run_id_is_idempotent(tmp_path) -> None:
    from alexdoor_xas.data_engine import run_baseline

    def run_once():
        return run_baseline(
            FakeDoorPushEnv(),
            outputs_root=tmp_path / "outputs",
            datasets_root=tmp_path / "datasets",
            experiment="idempotence",
            run_id="run",
            n_fixed=1,
            n_randomized=1,
            base_seed=0,
        )

    first = run_once()
    second = run_once()
    assert first.run_dir == second.run_dir

    episode_files = sorted((second.run_dir / "episodes").iterdir())
    # 2 episodes x (hdf5 + meta.json sidecar): a rerun replaces, never accumulates.
    assert len(episode_files) == 4
    metrics = json.loads((second.run_dir / "metrics" / "metrics.json").read_text())
    assert len(metrics["episodes"]) == 2


def test_export_rejects_empty_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty episode list"):
        export_datasets([], tmp_path)


@requires_h5py
def test_reexport_replaces_the_version_dir(tmp_path) -> None:
    export_datasets([_generate(seed=0)], tmp_path, version="v0")
    exported = export_datasets([_generate(seed=1)], tmp_path, version="v0")

    a2_files = sorted(exported[A2_EE_DELTA].glob("episode_*.hdf5"))
    assert len(a2_files) == 1
    assert read_episode(a2_files[0]).meta.seed == 1
    a4_lines = (exported[A4_OBJ_CENTRIC_CHUNK] / "episodes.jsonl").read_text().splitlines()
    assert len(a4_lines) == 1


# -- Local post-Phase 3.3 stabilization: door-pose obs, sanity, no-export --


def test_door_pose_obs_terms_recorded_and_round_trip(tmp_path) -> None:
    yaw = 0.7
    origin = (1.0, -2.0, 0.5)
    episode = run_episode(
        FakeDoorPushEnv(yaw_rad=yaw, origin=origin),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(door_pose_id="D3", door_yaw_rad=yaw),
    )

    step = episode.steps[0]
    assert step.object_state["door_yaw_rad"] == pytest.approx(yaw)
    # No robot_base_pos_w on the proxy fake -> relative to the world origin.
    assert step.object_state["door_rel_pos_x"] == pytest.approx(origin[0])
    assert step.object_state["door_rel_pos_y"] == pytest.approx(origin[1])
    assert step.object_state["door_rel_pos_z"] == pytest.approx(origin[2])
    assert episode.extras["door_pose_id"] == "D3"
    assert episode.extras["engine_cfg"]["door_pose_id"] == "D3"
    assert episode.extras["engine_cfg"]["door_yaw_rad"] == pytest.approx(yaw)

    if importlib.util.find_spec("h5py") is None:
        pytest.skip("h5py is not installed")
    from alexdoor_xas.recording import write_episode

    path = write_episode(episode, tmp_path)
    loaded = read_episode(path)
    loaded_step = loaded.steps[0]
    assert loaded_step.object_state["door_yaw_rad"] == pytest.approx(yaw)
    assert loaded_step.object_state["door_rel_pos_y"] == pytest.approx(origin[1])
    assert loaded.extras["door_pose_id"] == "D3"


@requires_h5py
def test_a2_a3_exports_differ_under_posed_door_and_match_rotation(tmp_path) -> None:
    """Problem-2 acceptance at unit level: yaw+translation separates A2 from A3.

    A3 must equal R_z(yaw)^T applied per-vector to A2 (deltas are free
    vectors: the door translation must not leak into the conversion), and at
    the default orientation the two exports stay numerically identical even
    with a translated door.
    """
    from alexdoor_xas.action.frames import rot_z

    yaw = 0.7
    episode = run_episode(
        FakeDoorPushEnv(yaw_rad=yaw, origin=(1.0, -2.0, 0.5)),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(),
    )
    exported = export_datasets([episode], tmp_path, version="vtest")
    a2 = np.stack(
        [s.action for s in read_episode(next(exported[A2_EE_DELTA].glob("*.hdf5"))).steps]
    )
    a3 = np.stack(
        [s.action for s in read_episode(next(exported[A3_OBJ_REL_EE_DELTA].glob("*.hdf5"))).steps]
    )
    assert np.abs(a2[:, :3]).max() > 0  # the episode actually moved
    assert np.abs(a2 - a3).max() > 1e-3  # numerically distinguishable
    rot = rot_z(yaw)
    np.testing.assert_allclose(a3[:, :3], a2[:, :3] @ rot, atol=1e-9)
    np.testing.assert_allclose(a3[:, 3:], a2[:, 3:] @ rot, atol=1e-9)

    # Default orientation + translated door: exports identical (rotation-only).
    episode0 = run_episode(
        FakeDoorPushEnv(yaw_rad=0.0, origin=(1.0, -2.0, 0.5)),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(),
    )
    exported0 = export_datasets([episode0], tmp_path / "id", version="vtest")
    b2 = np.stack(
        [s.action for s in read_episode(next(exported0[A2_EE_DELTA].glob("*.hdf5"))).steps]
    )
    b3 = np.stack(
        [s.action for s in read_episode(next(exported0[A3_OBJ_REL_EE_DELTA].glob("*.hdf5"))).steps]
    )
    np.testing.assert_allclose(b2, b3, atol=1e-12)


@requires_h5py
def test_run_baseline_writes_sanity_summary_and_respects_no_export(tmp_path) -> None:
    from alexdoor_xas.data_engine import run_baseline

    artifacts = run_baseline(
        FakeForceDoorPushEnv(),
        outputs_root=tmp_path / "outputs",
        datasets_root=tmp_path / "datasets",
        experiment="sanity",
        run_id="run",
        n_fixed=1,
        n_randomized=0,
        base_seed=0,
        export=False,
    )
    assert artifacts.exports == {}
    assert not (tmp_path / "datasets").exists()
    assert artifacts.sanity is not None
    assert artifacts.sanity["n_episodes_checked"] == 1
    assert artifacts.sanity["n_episodes_with_errors"] == 0
    summary = json.loads((artifacts.run_dir / "metrics" / "sanity.json").read_text())
    assert summary["episodes"][0]["seed"] == 0
    assert summary["episodes"][0]["errors"] == []


@requires_h5py
def test_run_baseline_aborts_loudly_before_export_on_sanity_error(tmp_path) -> None:
    from alexdoor_xas.data_engine import run_baseline

    class WindupEnv(FakeForceDoorPushEnv):
        """Joint targets march far past tight limits -> hard sanity error."""

        def robot_joint_limits(self):
            limits = super().robot_joint_limits()
            limits["joint_pos_limits"] = np.stack(
                [np.full(self.N_JOINTS, -0.01), np.full(self.N_JOINTS, 0.01)], axis=1
            )
            return limits

    with pytest.raises(RuntimeError, match="sanity checks failed"):
        run_baseline(
            WindupEnv(),
            outputs_root=tmp_path / "outputs",
            datasets_root=tmp_path / "datasets",
            experiment="sanity",
            run_id="bad",
            n_fixed=1,
            n_randomized=0,
            base_seed=0,
        )
    # The summary is written for debugging, but nothing reached datasets/.
    summary_path = tmp_path / "outputs" / "sanity" / "bad" / "metrics" / "sanity.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["n_episodes_with_errors"] == 1
    assert not (tmp_path / "datasets").exists()


@requires_h5py
def test_run_baseline_aborts_before_export_on_force_admission_error(tmp_path) -> None:
    from alexdoor_xas.data_engine import run_baseline

    class ForceSpikeEnv(FakeForceDoorPushEnv):
        def _contact_force_n(self) -> float:
            if self._ticks == 3:
                return 500.0
            return super()._contact_force_n()

    with pytest.raises(RuntimeError, match="sanity checks failed"):
        run_baseline(
            ForceSpikeEnv(),
            outputs_root=tmp_path / "outputs",
            datasets_root=tmp_path / "datasets",
            experiment="force_gate",
            run_id="bad",
            n_fixed=1,
            n_randomized=0,
            base_seed=0,
            export=True,
        )

    summary_path = tmp_path / "outputs/force_gate/bad/metrics/sanity.json"
    summary = json.loads(summary_path.read_text())
    assert any(
        "force admission limit" in error for error in summary["episodes"][0]["errors"]
    )
    assert not (tmp_path / "datasets").exists()


@requires_h5py
def test_run_baseline_aborts_before_export_on_negative_force(tmp_path, monkeypatch) -> None:
    import dataclasses

    from alexdoor_xas.data_engine import run_baseline
    from alexdoor_xas.data_engine import runner as runner_module

    original_run_episode = runner_module.run_episode

    def run_episode_with_negative_force(*args, **kwargs):
        episode = original_run_episode(*args, **kwargs)
        contact = dict(episode.steps[3].contact)
        contact["force_n"] = -1.0
        episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)
        return episode

    monkeypatch.setattr(runner_module, "run_episode", run_episode_with_negative_force)

    with pytest.raises(RuntimeError, match="sanity checks failed"):
        run_baseline(
            FakeForceDoorPushEnv(),
            outputs_root=tmp_path / "outputs",
            datasets_root=tmp_path / "datasets",
            experiment="negative_force_gate",
            run_id="bad",
            n_fixed=1,
            n_randomized=0,
            base_seed=0,
            export=True,
        )

    summary_path = tmp_path / "outputs/negative_force_gate/bad/metrics/sanity.json"
    summary = json.loads(summary_path.read_text())
    entry = summary["episodes"][0]
    assert any("force magnitude must be non-negative" in error for error in entry["errors"])
    assert entry["force_diagnostics"]["min_force_n"] == -1.0
    assert entry["force_diagnostics"]["negative_force_ticks"] == [3]
    assert entry["force_diagnostics"]["force_admission_passed"] is False
    assert not (tmp_path / "datasets").exists()


@requires_h5py
def test_run_baseline_refuses_direct_export_from_posed_runs(tmp_path) -> None:
    """A posed run with export enabled would replace the official default-pose
    dataset version; only the merged-export script may write multi-pose data."""
    from alexdoor_xas.data_engine import run_baseline

    with pytest.raises(RuntimeError, match="non-default door pose"):
        run_baseline(
            FakeForceDoorPushEnv(),
            outputs_root=tmp_path / "outputs",
            datasets_root=tmp_path / "datasets",
            experiment="pose_guard",
            run_id="bad",
            n_fixed=1,
            n_randomized=0,
            base_seed=0,
            engine_cfg=DataEngineCfg(door_pose_id="D1", door_yaw_rad=0.05),
        )
    assert not (tmp_path / "datasets").exists()
    # export=False stays allowed for posed runs (the multi-pose flow).
    artifacts = run_baseline(
        FakeForceDoorPushEnv(),
        outputs_root=tmp_path / "outputs",
        datasets_root=tmp_path / "datasets",
        experiment="pose_guard",
        run_id="ok",
        n_fixed=1,
        n_randomized=0,
        base_seed=0,
        engine_cfg=DataEngineCfg(door_pose_id="D1", door_yaw_rad=0.05),
        export=False,
    )
    assert artifacts.exports == {}
    # Config provenance survives even for runs that would later abort.
    assert (artifacts.run_dir / "logs" / "run_config.json").exists()
