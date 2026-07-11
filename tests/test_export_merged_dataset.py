"""Regression tests for the official merged-dataset admission boundary."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.recording import write_episode
from conftest import FakeForceDoorPushEnv


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_merged_dataset.py"
    spec = importlib.util.spec_from_file_location("export_merged_dataset_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merged_export_rechecks_force_limit_instead_of_trusting_stored_sanity() -> None:
    episode = run_episode(
        FakeForceDoorPushEnv(),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(door_pose_id="D0"),
    )
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = 500.0
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)
    force_summary = _module().episode_contact_safety(episode)
    assert force_summary["force_admission"] == {
        "limit_n": 200.0,
        "passed": False,
        "all_forces_finite": True,
        "non_finite_force_ticks": [],
        "min_force_n": 0.0,
        "min_force_tick": 0,
        "negative_force_ticks": [],
        "max_force_n": 500.0,
        "ticks_over_limit": [3],
    }
    pose = {"pose_id": "D0", "base_seed": 0, "overdraw_base_seed": 500}
    plan = {
        "seed_block_size": 50,
        "episodes_fixed_per_pose": 1,
        "episodes_randomized_per_pose": 0,
        "poses": [pose],
    }
    runs = {
        "D0": {
            "pose": pose,
            "episodes": [episode],
            "overdraw_episodes": [],
        }
    }

    with pytest.raises(RuntimeError, match="force admission limit"):
        _module().verify_merged(plan, runs)


def test_merged_export_rejects_negative_force_reloaded_from_stored_episode(
    tmp_path,
) -> None:
    pytest.importorskip("h5py")
    module = _module()
    episode = run_episode(
        FakeForceDoorPushEnv(),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(door_pose_id="D0"),
    )
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = -1.5
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)

    run_dir = tmp_path / "outputs" / "negative_force" / "poseD0"
    write_episode(episode, run_dir / "episodes")
    sanity_dir = run_dir / "metrics"
    sanity_dir.mkdir(parents=True)
    (sanity_dir / "sanity.json").write_text(
        json.dumps(
            {
                "n_episodes_checked": 1,
                "n_episodes_with_errors": 0,
                "n_episodes_with_warnings": 0,
                "episodes": [],
            }
        )
    )
    pose = {"pose_id": "D0", "base_seed": 0, "overdraw_base_seed": 500}
    plan = {
        "seed_block_size": 50,
        "episodes_fixed_per_pose": 1,
        "episodes_randomized_per_pose": 0,
        "poses": [pose],
    }

    runs = module.load_pose_runs(plan, tmp_path / "outputs", "negative_force")
    assert runs["D0"]["episodes"][0].steps[3].contact["force_n"] == -1.5
    with pytest.raises(RuntimeError, match="force magnitude must be non-negative"):
        module.verify_merged(plan, runs)


def test_manifest_force_admission_uses_shared_negative_force_semantics() -> None:
    episode = run_episode(
        FakeForceDoorPushEnv(),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(door_pose_id="D0"),
    )
    contact = dict(episode.steps[3].contact)
    contact["force_n"] = -0.25
    episode.steps[3] = dataclasses.replace(episode.steps[3], contact=contact)

    admission = _module().episode_contact_safety(episode)["force_admission"]
    assert admission["passed"] is False
    assert admission["all_forces_finite"] is True
    assert admission["min_force_n"] == -0.25
    assert admission["min_force_tick"] == 3
    assert admission["negative_force_ticks"] == [3]
    assert admission["max_force_n"] <= 200.0
    assert admission["ticks_over_limit"] == []
