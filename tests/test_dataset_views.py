"""Regression tests for directly loaded retained dataset views."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.data_engine import DataEngineCfg, export_datasets, plan_episodes, run_episode
from alexdoor_xas.dataset import (
    EpisodeDataset,
    compute_norm_stats,
    load_view_payload,
    save_norm_stats,
    view_norm_stats_path,
)
from alexdoor_xas.policies.act.checkpoint import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.common.data import PolicyDataError, load_policy_data
from conftest import FakeForceDoorPushEnv


def _dataset(tmp_path: Path) -> tuple[Path, EpisodeDataset]:
    episodes = [
        run_episode(
            FakeForceDoorPushEnv(start_door_frame=(0.7, 0.2 + index * 0.01, 0.0)),
            plan_episodes(0, 1, index)[0],
            DataEngineCfg(task="door_push", door_pose_id="D0"),
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
    save_norm_stats(
        view_norm_stats_path(dataset.dataset_dir, "view_n2"),
        stats,
    )


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        task="door_push",
        space="A2_ee_delta",
        version="master",
        view_id="view_n2",
        obs_preset="core_door_pose",
    )


def test_view_loader_tolerates_legacy_extra_fields(tmp_path) -> None:
    path = tmp_path / "view.json"
    path.write_text(
        json.dumps(
            {
                "schema": "alexdoor_xas.dataset_view.v1",
                "view_id": "view_n2",
                "master_version": "master",
                "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
                "obsolete_fingerprint": "ignored",
            }
        )
    )
    assert load_view_payload(path)["splits"]["train"] == ["a"]


def test_policy_data_resolves_view_split_and_recomputes_stats(tmp_path) -> None:
    root, dataset = _dataset(tmp_path)
    _write_view(root, dataset)

    data = load_policy_data(_cfg(), root)

    assert len(data.train_ids) == 2
    assert len(data.val_ids) == 1
    assert len(data.test_ids) == 1
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


def test_view_checkpoint_v2_round_trip_needs_no_split_provenance(tmp_path) -> None:
    root, dataset = _dataset(tmp_path)
    _write_view(root, dataset)
    data = load_policy_data(_cfg(), root)
    model = ACTModel(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        cfg=ActModelCfg(
            chunk_size=2,
            d_model=16,
            n_heads=2,
            dim_feedforward=32,
            z_dim=4,
            cvae_encoder_layers=1,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0,
        ),
    )
    path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        {"dataset": vars(_cfg())},
        data.stats,
    )

    loaded = load_checkpoint(path)

    assert loaded.checkpoint_format == CHECKPOINT_FORMAT
    assert loaded.config == {"dataset": vars(_cfg())}
    assert loaded.stats.view_id == "view_n2"
