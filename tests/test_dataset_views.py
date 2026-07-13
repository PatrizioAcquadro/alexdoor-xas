"""Regression contract for nested, fixed-holdout dataset views."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from alexdoor_xas.data_engine import DataEngineCfg, export_datasets, plan_episodes, run_episode
from alexdoor_xas.dataset import (
    EpisodeDataset,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
    split_fingerprint,
    validate_norm_stats,
)
from alexdoor_xas.dataset.splits import SplitEntry
from alexdoor_xas.dataset.views import (
    VIEW_SCHEMA,
    build_nested_views,
    load_view_payload,
    save_view_payload,
    validate_nested_views,
    view_fingerprint,
)
from alexdoor_xas.policies.act.checkpoint import load_checkpoint, save_checkpoint
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.common.data import PolicyData, checkpoint_provenance
from conftest import FakeForceDoorPushEnv

POSES = ("D0", "D1", "D2", "D3", "D4")
VIEW_TRAIN_COUNTS = {
    "v3_scale_n50": 50,
    "v3_scale_n100": 100,
    "v3_scale_n250": 250,
    "v3_scale_n500": 500,
}


def _master_entries() -> list[SplitEntry]:
    return [
        SplitEntry(
            episode_id=f"{pose}-episode-{index:03d}",
            group_key=f"{pose}-content-{index:03d}",
            pose_id=pose,
        )
        for pose in POSES
        for index in range(110)
    ]


def test_nested_views_freeze_exact_counts_balance_holdouts_and_nesting() -> None:
    views = build_nested_views(
        _master_entries(),
        view_train_counts=VIEW_TRAIN_COUNTS,
        pose_ids=POSES,
        seed=3407,
        master_version="v3_scale_master",
        master_fingerprint="a" * 64,
    )
    assert set(views) == set(VIEW_TRAIN_COUNTS)
    assert validate_nested_views(
        views,
        _master_entries(),
        view_train_counts=VIEW_TRAIN_COUNTS,
        pose_ids=POSES,
        master_version="v3_scale_master",
        master_fingerprint="a" * 64,
    ) == []

    fixed_val = views["v3_scale_n50"]["splits"]["val"]
    fixed_test = views["v3_scale_n50"]["splits"]["test"]
    previous_train: set[str] = set()
    for view_id, n_train in VIEW_TRAIN_COUNTS.items():
        payload = views[view_id]
        splits = payload["splits"]
        assert payload["schema"] == VIEW_SCHEMA
        assert payload["view_id"] == view_id
        assert len(splits["train"]) == n_train
        assert len(splits["val"]) == 25
        assert len(splits["test"]) == 25
        assert splits["val"] == fixed_val
        assert splits["test"] == fixed_test
        assert previous_train < set(splits["train"]) if previous_train else True
        previous_train = set(splits["train"])

        membership = {
            entry.episode_id: entry.pose_id for entry in _master_entries()
        }
        per_pose_train = {
            pose: sum(membership[episode_id] == pose for episode_id in splits["train"])
            for pose in POSES
        }
        per_pose_val = {
            pose: sum(membership[episode_id] == pose for episode_id in splits["val"])
            for pose in POSES
        }
        per_pose_test = {
            pose: sum(membership[episode_id] == pose for episode_id in splits["test"])
            for pose in POSES
        }
        assert set(per_pose_train.values()) == {n_train // len(POSES)}
        assert set(per_pose_val.values()) == {5}
        assert set(per_pose_test.values()) == {5}
        assert set(splits["train"]).isdisjoint(splits["val"])
        assert set(splits["train"]).isdisjoint(splits["test"])
        assert set(splits["val"]).isdisjoint(splits["test"])


def test_nested_views_are_deterministic_and_seed_bound() -> None:
    kwargs = {
        "view_train_counts": VIEW_TRAIN_COUNTS,
        "pose_ids": POSES,
        "seed": 3407,
        "master_version": "v3_scale_master",
        "master_fingerprint": "b" * 64,
    }
    first = build_nested_views(_master_entries(), **kwargs)
    second = build_nested_views(_master_entries(), **kwargs)
    changed = build_nested_views(_master_entries(), **{**kwargs, "seed": 3408})
    assert first == second
    assert first != changed
    assert {
        key: view_fingerprint(value) for key, value in first.items()
    } == {
        key: value["view_fingerprint_sha256"] for key, value in first.items()
    }


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda entries: entries[:-1], "110"),
        (
            lambda entries: [
                *entries[:-1],
                SplitEntry("replacement", entries[0].group_key, "D4"),
            ],
            "content",
        ),
        (
            lambda entries: [
                *entries,
                SplitEntry(entries[0].episode_id, "new-content", "D0"),
            ],
            "episode ids",
        ),
    ],
)
def test_view_builder_rejects_incomplete_duplicate_or_non_distinct_master(
    mutator, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_nested_views(
            mutator(_master_entries()),
            view_train_counts=VIEW_TRAIN_COUNTS,
            pose_ids=POSES,
            seed=3407,
            master_version="v3_scale_master",
            master_fingerprint="c" * 64,
        )


def test_view_validation_rejects_holdout_drift_non_nesting_and_group_leakage() -> None:
    views = build_nested_views(
        _master_entries(),
        view_train_counts=VIEW_TRAIN_COUNTS,
        pose_ids=POSES,
        seed=3407,
        master_version="v3_scale_master",
        master_fingerprint="d" * 64,
    )
    drifted = copy.deepcopy(views)
    drifted["v3_scale_n100"]["splits"]["val"][0] = drifted["v3_scale_n100"][
        "splits"
    ]["train"][0]
    failures = validate_nested_views(
        drifted,
        _master_entries(),
        view_train_counts=VIEW_TRAIN_COUNTS,
        pose_ids=POSES,
        master_version="v3_scale_master",
        master_fingerprint="d" * 64,
    )
    assert any("holdout" in failure or "overlap" in failure for failure in failures)


def test_view_payload_round_trip_rejects_tampering(tmp_path) -> None:
    payload = build_nested_views(
        _master_entries(),
        view_train_counts=VIEW_TRAIN_COUNTS,
        pose_ids=POSES,
        seed=3407,
        master_version="v3_scale_master",
        master_fingerprint="e" * 64,
    )["v3_scale_n50"]
    path = save_view_payload(tmp_path / "v3_scale_n50.json", payload)
    assert load_view_payload(path) == payload

    tampered = json.loads(path.read_text())
    tampered["splits"]["train"][0] = "tampered"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint"):
        load_view_payload(path)


def _tiny_view_dataset(tmp_path: Path) -> EpisodeDataset:
    episodes = [
        run_episode(
            FakeForceDoorPushEnv(start_door_frame=(0.7, 0.2 + index * 0.01, 0.0)),
            plan_episodes(0, 1, index)[0],
            DataEngineCfg(task="door_push", door_pose_id="D0"),
        )
        for index in range(4)
    ]
    exported = export_datasets(episodes, tmp_path / "datasets", version="master")
    return EpisodeDataset(exported["A2_ee_delta"])


def test_view_normalization_is_train_only_and_hash_bound(tmp_path) -> None:
    dataset = _tiny_view_dataset(tmp_path)
    train_ids = dataset.episode_ids[:2]
    stats = compute_norm_stats(
        dataset,
        train_ids,
        "core_door_pose",
        view_id="view_n2",
        view_fingerprint="f" * 64,
    )
    path = save_norm_stats(tmp_path / "norm_stats.json", stats)
    loaded = load_norm_stats(path)
    assert loaded.train_episode_ids == tuple(train_ids)
    assert loaded.action.count == sum(dataset.by_id(item).n_steps for item in train_ids)
    assert loaded.view_id == "view_n2"
    assert loaded.view_fingerprint == "f" * 64
    assert len(loaded.normalization_fingerprint) == 64
    assert validate_norm_stats(
        loaded,
        dataset,
        train_ids,
        "core_door_pose",
        view_id="view_n2",
        view_fingerprint="f" * 64,
    ) == []
    assert validate_norm_stats(
        loaded,
        dataset,
        dataset.episode_ids[:3],
        "core_door_pose",
        view_id="view_n2",
        view_fingerprint="f" * 64,
    )
    assert validate_norm_stats(
        replace(loaded, view_fingerprint="0" * 64),
        dataset,
        train_ids,
        "core_door_pose",
        view_id="view_n2",
        view_fingerprint="f" * 64,
    )


def test_view_normalization_recomputes_values_even_after_hash_refresh(tmp_path) -> None:
    dataset = _tiny_view_dataset(tmp_path)
    train_ids = dataset.episode_ids[:2]
    path = save_norm_stats(
        tmp_path / "norm_stats.json",
        compute_norm_stats(
            dataset,
            train_ids,
            "core_door_pose",
            view_id="view_n2",
            view_fingerprint="f" * 64,
        ),
    )
    payload = json.loads(path.read_text())
    payload["action"]["mean"][0] += 123.0
    payload["normalization_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "normalization_fingerprint_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    failures = validate_norm_stats(
        load_norm_stats(path),
        dataset,
        train_ids,
        "core_door_pose",
        view_id="view_n2",
        view_fingerprint="f" * 64,
    )
    assert any("recomputed action mean" in failure for failure in failures)


def test_view_checkpoint_requires_and_round_trips_full_training_provenance(tmp_path) -> None:
    dataset = _tiny_view_dataset(tmp_path)
    train_ids = tuple(dataset.episode_ids[:2])
    stats_path = save_norm_stats(
        tmp_path / "view_norm.json",
        compute_norm_stats(
            dataset,
            list(train_ids),
            "core_door_pose",
            view_id="view_n2",
            view_fingerprint="e" * 64,
        ),
    )
    stats = load_norm_stats(stats_path)
    data = PolicyData(
        dataset=dataset,
        train_ids=train_ids,
        val_ids=(dataset.episode_ids[2],),
        test_ids=(dataset.episode_ids[3],),
        stats=stats,
        stats_source="official",
        robot_asset=None,
        robot_asset_manifest=None,
        view_id="view_n2",
        view_fingerprint="e" * 64,
        split_fingerprint=split_fingerprint(
            {
                "train": list(train_ids),
                "val": [dataset.episode_ids[2]],
                "test": [dataset.episode_ids[3]],
            }
        ),
        stats_path=stats_path,
        stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        master_dataset_fingerprint="f" * 64,
        action_dataset_fingerprint=stats.dataset_fingerprint,
    )
    config = {
        "dataset": {
            "task": "door_push",
            "space": "A2_ee_delta",
            "version": "master",
            "view_id": "view_n2",
            "obs_preset": "core_door_pose",
        },
        "train": {"seed": 0},
    }
    provenance = checkpoint_provenance(data, config, source_git_commit="1" * 40)
    assert provenance["master_dataset_fingerprint_sha256"] == "f" * 64
    assert provenance["action_dataset_fingerprint_sha256"] == stats.dataset_fingerprint
    assert (
        provenance["master_dataset_fingerprint_sha256"]
        != provenance["action_dataset_fingerprint_sha256"]
    )
    model = ACTModel(
        obs_dim=stats.obs.dim,
        action_dim=stats.action.dim,
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
    split_ids = {"train": train_ids, "val": data.val_ids, "test": data.test_ids}
    path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        config,
        stats,
        split_episode_ids=split_ids,
        provenance=provenance,
    )
    loaded = load_checkpoint(path)
    assert loaded.provenance == provenance
    assert loaded.stats.view_id == "view_n2"

    with pytest.raises(ValueError, match="training provenance"):
        save_checkpoint(
            tmp_path / "missing.pt",
            model,
            config,
            stats,
            split_episode_ids=split_ids,
        )
    with pytest.raises(ValueError, match="normalization_sha256"):
        save_checkpoint(
            tmp_path / "tampered.pt",
            model,
            config,
            stats,
            split_episode_ids=split_ids,
            provenance={**provenance, "normalization_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="action_dataset_fingerprint_sha256"):
        save_checkpoint(
            tmp_path / "tampered-action-fingerprint.pt",
            model,
            config,
            stats,
            split_episode_ids=split_ids,
            provenance={**provenance, "action_dataset_fingerprint_sha256": "0" * 64},
        )
