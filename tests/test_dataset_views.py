"""Tests for retained dataset views and their train-only normalization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.data_engine import export_datasets, plan_episodes, run_episode
from alexdoor_xas.dataset.loader import EpisodeDataset
from alexdoor_xas.dataset.normalize import (
    compute_norm_stats,
    save_norm_stats,
    view_norm_stats_path,
)
from alexdoor_xas.policies.common.data import PolicyDataError, load_policy_data
from conftest import FakeForceDoorPushEnv, make_test_engine_cfg


def _dataset(tmp_path: Path) -> tuple[Path, EpisodeDataset]:
    episodes = [
        run_episode(
            FakeForceDoorPushEnv(start_door_frame=(0.7, 0.2 + index * 0.01, 0.0)),
            plan_episodes(0, 1, index)[0],
            make_test_engine_cfg(task="door_push", door_pose_id="D0"),
        )
        for index in range(4)
    ]
    root = tmp_path / "datasets"
    exported = export_datasets(episodes, root, version="master")
    return root, EpisodeDataset(exported["A2_ee_delta"])


def _write_view(root: Path, dataset: EpisodeDataset, *, overlap: bool = False) -> None:
    train = dataset.episode_ids[:2]
    val = [dataset.episode_ids[1 if overlap else 2]]
    test = [dataset.episode_ids[3]]
    split_path = root / "door_push" / "splits" / "view_n2.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "schema": "alexdoor_xas.dataset_view.v1",
                "view_id": "view_n2",
                "master_version": "master",
                "splits": {"train": train, "val": val, "test": test},
                "view_fingerprint_sha256": "legacy-extra-field-is-ignored",
            }
        )
    )
    stats = compute_norm_stats(dataset, train, obs_preset="core_door_pose", view_id="view_n2")
    save_norm_stats(view_norm_stats_path(dataset.dataset_dir, "view_n2"), stats)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        task="door_push",
        space="A2_ee_delta",
        version="master",
        view_id="view_n2",
        obs_preset="core_door_pose",
    )


def test_policy_data_loads_a_valid_view(tmp_path) -> None:
    root, dataset = _dataset(tmp_path)
    _write_view(root, dataset)

    data = load_policy_data(_cfg(), root)

    assert tuple(map(len, (data.train_ids, data.val_ids, data.test_ids))) == (2, 1, 1)
    assert data.stats.view_id == "view_n2"
    assert data.stats_source == "official"


def test_policy_data_rejects_overlap_and_stale_stats(tmp_path) -> None:
    root, dataset = _dataset(tmp_path)
    _write_view(root, dataset, overlap=True)
    with pytest.raises(PolicyDataError, match="overlapping"):
        load_policy_data(_cfg(), root)

    _write_view(root, dataset)
    stats_path = view_norm_stats_path(dataset.dataset_dir, "view_n2")
    payload = json.loads(stats_path.read_text())
    payload["action"]["mean"][0] += 1.0
    stats_path.write_text(json.dumps(payload))
    with pytest.raises(PolicyDataError, match="recomputed action mean"):
        load_policy_data(_cfg(), root)
