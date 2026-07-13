"""Synthetic regressions for resumable scale-master selection/publication."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.recording import write_episode
from conftest import FakeForceDoorPushEnv

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scale_module():
    path = REPO_ROOT / "scripts" / "build_scale_dataset.py"
    spec = importlib.util.spec_from_file_location("build_scale_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_scale_plan_is_strict_randomized_and_seed_disjoint(tmp_path) -> None:
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    source = REPO_ROOT / "configs/door_pose_plan_v3_scale.json"
    plan = scale._load_plan(source, config)
    assert plan["fixed_candidates_per_pose"] == 0
    assert plan["randomized_candidates_only"] is True
    assert plan["source_candidates_per_pose"] == 110
    assert plan["selected_episodes_per_pose"] == 110
    assert plan["overdraw_candidates_per_pose"] > 0

    drifted = json.loads(source.read_text())
    drifted["poses"][1]["overdraw_seed_start"] = drifted["poses"][0][
        "source_seed_start"
    ]
    drifted["poses"][1]["overdraw_seed_stop"] = (
        drifted["poses"][1]["overdraw_seed_start"]
        + drifted["overdraw_candidates_per_pose"]
    )
    bad = tmp_path / "overlap.json"
    bad.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="overlap"):
        scale._load_plan(bad, config)


def _candidate_fixture(tmp_path: Path):
    run = tmp_path / "poseD0_attempt001"
    episodes = run / "episodes"
    for seed in (10, 11, 20):
        episode = run_episode(
            FakeForceDoorPushEnv(
                start_door_frame=(0.7, 0.2 + 0.001 * seed, 0.0)
            ),
            plan_episodes(0, 1, seed)[0],
            DataEngineCfg(task="door_push", door_pose_id="D0"),
        )
        if seed == 10:
            assert episode.outcome is not None
            episode.outcome = replace(
                episode.outcome,
                success=False,
                failure_label="synthetic_source_failure",
            )
        write_episode(episode, episodes)
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "sanity.json").write_text(
        json.dumps({"n_episodes_checked": 3}) + "\n"
    )
    plan = {
        "selected_episodes_per_pose": 2,
        "poses": [
            {
                "pose_id": "D0",
                "source_seed_start": 10,
                "source_seed_stop": 12,
                "overdraw_seed_start": 20,
                "overdraw_seed_stop": 21,
            }
        ],
    }
    state = {"poses": {"D0": {"completed": str(run)}}}
    return plan, state


def test_master_selection_uses_overdraw_only_for_failed_source_and_records_provenance(
    tmp_path,
) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    assert [episode.meta.seed for episode in selected] == [11, 20]
    assert len(paths) == 2
    rows = {row["seed"]: row for row in provenance}
    assert rows[10]["decision"] == "SKIPPED"
    assert rows[10]["namespace"] == "source"
    assert any("task failure" in reason for reason in rows[10]["reasons"])
    assert rows[11]["decision"] == "SELECTED"
    assert rows[20]["decision"] == "SELECTED"
    assert rows[20]["namespace"] == "overdraw"
    assert rows[20]["replacement_for_seed"] == 10
    assert all(episode.extras["variation"] is not None for episode in selected)


def test_master_selection_fails_closed_when_overdraw_cannot_fill_quota(tmp_path) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    plan["selected_episodes_per_pose"] = 3
    with pytest.raises(RuntimeError, match="need 3"):
        scale._select_master(plan, state)
