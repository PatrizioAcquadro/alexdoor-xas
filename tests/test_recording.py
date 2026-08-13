"""Pure tests for the episode buffer and the HDF5 + JSON episode container."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.recording import (
    EpisodeBuffer,
    EpisodeMeta,
    EpisodeOutcome,
    EpisodeStep,
    read_episode,
    write_episode,
)

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)


def _make_episode(n_steps: int = 3) -> EpisodeBuffer:
    meta = EpisodeMeta.create(
        task="door_push",
        action_space=A2_EE_DELTA,
        robot="synthetic_test_double",
        scene="outputs/door_scene/D0.usda",
        policy="scripted",
        seed=7,
        sim_dt=1 / 120,
        control_dt=1 / 60,
        robot_asset_id="synthetic",
        robot_asset_sha256="a" * 64,
    )
    buffer = EpisodeBuffer(meta=meta)
    for i in range(n_steps):
        buffer.add_step(
            EpisodeStep(
                t=i / 60,
                action=np.full(6, 0.001 * (i + 1)),
                proprio={
                    "ee_pos_w": np.array([0.1 * i, 0.2, 1.0]),
                    "ee_quat_w_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                },
                object_state={
                    "door_angle_rad": 0.1 * i,
                    "door_angular_velocity_rad_s": 0.05,
                },
                contact={"inferred": i == 2, "source": "inferred_geometric"},
                safety={"controller_phase": "approach"},
            )
        )
    buffer.extras = {
        "action_door_frame": np.full((n_steps, 6), 0.002),
        "door_frame_pos_w": np.array([-0.7, -0.45, 1.01]),
        "a4_chunks": [{"phase": "approach", "duration_ticks": n_steps}],
        "variation": None,
    }
    buffer.set_outcome(
        EpisodeOutcome(
            success=True,
            final_door_angle=0.9,
            n_steps=n_steps,
            termination_reason="controller_done",
            environment_terminated=False,
            environment_truncated=False,
        )
    )
    return buffer


@requires_h5py
def test_zero_step_episode_round_trip(tmp_path) -> None:
    import h5py

    buffer = _make_episode(0)
    buffer.outcome = EpisodeOutcome(
        success=False,
        final_door_angle=0.0,
        n_steps=0,
        termination_reason="tick_budget",
        environment_terminated=False,
        environment_truncated=False,
    )

    path = write_episode(buffer, tmp_path)
    with h5py.File(path, "r") as h5:
        assert h5["steps/action"].shape == (0, 6)

    loaded = read_episode(path)
    assert loaded.n_steps == 0
    assert loaded.steps == []
    assert loaded.meta == buffer.meta
    assert loaded.outcome == buffer.outcome


def test_add_step_rejects_bad_action_shape() -> None:
    buffer = _make_episode(0)
    buffer.outcome = None
    step = EpisodeStep(
        t=0.0, action=np.zeros(3), proprio={}, object_state={}, contact={}, safety={}
    )
    with pytest.raises(ValueError, match="action must have shape"):
        buffer.add_step(step)


def test_outcome_step_count_must_match() -> None:
    buffer = _make_episode(2)
    buffer.outcome = None
    with pytest.raises(ValueError, match="does not match"):
        buffer.set_outcome(
            EpisodeOutcome(
                success=False,
                final_door_angle=0.0,
                n_steps=5,
                termination_reason="tick_budget",
                environment_terminated=False,
                environment_truncated=False,
            )
        )


@requires_h5py
def test_episode_round_trip(tmp_path) -> None:
    import h5py

    original = _make_episode()
    h5_path = write_episode(original, tmp_path)
    with h5py.File(h5_path, "r") as h5:
        assert "chunk_len" not in h5["meta"].attrs
        assert "obs_ref" not in h5["steps"]
        assert set(h5["steps/safety"]) == {"controller_phase"}
    assert not h5_path.with_suffix(".meta.json").exists()

    loaded = read_episode(h5_path)

    assert loaded.meta == original.meta
    assert loaded.outcome == original.outcome
    assert loaded.n_steps == original.n_steps
    for i, (a, b) in enumerate(zip(loaded.steps, original.steps, strict=True)):
        assert a.t == pytest.approx(b.t)
        np.testing.assert_allclose(a.action, b.action)
        np.testing.assert_allclose(a.proprio["ee_pos_w"], b.proprio["ee_pos_w"])
        assert a.contact["inferred"] == b.contact["inferred"], f"step {i}"
        assert a.contact["source"] == "inferred_geometric"
        assert a.safety["controller_phase"] == "approach"
    np.testing.assert_allclose(
        loaded.extras["action_door_frame"], original.extras["action_door_frame"]
    )
    assert loaded.extras["a4_chunks"] == original.extras["a4_chunks"]
    assert loaded.extras["variation"] is None


@requires_h5py
def test_legacy_episode_reads_without_failure_label(tmp_path) -> None:
    import h5py

    path = write_episode(_make_episode(), tmp_path)
    with h5py.File(path, "r+") as h5:
        h5.attrs["schema_version"] = "phase2.v1"
        h5["meta"].attrs["chunk_len"] = 1
        obs_ref = h5["steps"].create_group("obs_ref")
        obs_ref.create_dataset("door_angle_rad", data=np.zeros(3))
        h5["steps/safety"].create_dataset("pos_clamped", data=np.zeros(3, dtype=bool))
        outcome = h5["outcome"].attrs
        del outcome["termination_reason"]
        del outcome["environment_terminated"]
        del outcome["environment_truncated"]
        outcome["failure_label"] = "obsolete_interpretation"

    loaded = read_episode(path)
    assert loaded.outcome.termination_reason == "not_recorded"
    assert loaded.outcome.environment_terminated is None
    assert loaded.outcome.environment_truncated is None
    assert "failure_label" not in loaded.outcome.to_dict()
    assert loaded.steps[0].safety == {"controller_phase": "approach"}


@requires_h5py
def test_write_requires_outcome(tmp_path) -> None:
    buffer = _make_episode()
    buffer.outcome = None
    with pytest.raises(ValueError, match="outcome must be set"):
        write_episode(buffer, tmp_path)
