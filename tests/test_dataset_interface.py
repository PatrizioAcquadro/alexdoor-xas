"""Tests for A1-A4 loading, validation, normalization, and batching."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
    A4_PHASE_VOCAB,
    EE_DELTA_DIM,
)
from alexdoor_xas.data_engine import export_datasets, plan_episodes, run_episode
from alexdoor_xas.dataset.loader import A4ChunkDataset, EpisodeDataset, obs_matrix
from alexdoor_xas.dataset.normalize import (
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path,
    save_norm_stats,
    validate_norm_stats,
)
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler
from alexdoor_xas.dataset.splits import splits_path
from alexdoor_xas.dataset.validate import (
    validate_a4_dataset,
    validate_dataset,
    validate_episode,
    validate_matched_action_space_datasets,
)
from conftest import FakeDoorPushEnv, make_test_engine_cfg

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)
pytestmark = requires_h5py

N_EPISODES = 4


def _copy_dataset(src: Path, tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def _jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _load_script(path: str):
    script_path = Path(__file__).resolve().parents[1] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export(tmp_root, env_factory):
    # Distinct start poses per seed: the fake env is deterministic, so equal
    # fixed episodes would collapse into one content-equivalence group and the
    # grouped split contract would (correctly) refuse to split them 3 ways.
    episodes = [
        run_episode(
            env_factory(start_door_frame=(0.7, 0.2 + 0.005 * seed, 0.0)),
            plan_episodes(1, 0, seed)[0],
            make_test_engine_cfg(),
        )
        for seed in range(N_EPISODES)
    ]
    return export_datasets(episodes, tmp_root, version="v0")


@pytest.fixture(scope="module")
def synthetic_exports(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("synthetic"), FakeDoorPushEnv)


@pytest.fixture(scope="module")
def alex_exports(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("alex"), FakeDoorPushEnv)


@pytest.fixture(scope="module")
def synthetic_a2(synthetic_exports) -> EpisodeDataset:
    return EpisodeDataset(synthetic_exports[A2_EE_DELTA])


@pytest.fixture(scope="module")
def alex_a2(alex_exports) -> EpisodeDataset:
    return EpisodeDataset(alex_exports[A2_EE_DELTA])


def test_dataset_loads_records_with_stacked_arrays(synthetic_a2) -> None:
    assert len(synthetic_a2) == N_EPISODES
    assert synthetic_a2.action_space == A2_EE_DELTA
    assert synthetic_a2.action_dim == EE_DELTA_DIM
    record = synthetic_a2[0]
    assert record.n_steps > 0
    assert record.actions.shape == (record.n_steps, EE_DELTA_DIM)
    assert record.t.shape == (record.n_steps,)
    assert record.schema_version == "phase2.v2"
    assert record.success and record.termination_reason == "controller_done"
    assert "failure_label" not in record.__dataclass_fields__
    assert {"ee_pos_w", "ee_quat_w_xyzw", "door_angle_rad", "inferred"} <= set(record.obs)
    assert synthetic_a2.by_id(record.episode_id) is record
    with pytest.raises(KeyError):
        synthetic_a2.by_id("no-such-episode")


def test_a1_dataset_has_joint_wide_actions(alex_exports) -> None:
    a1 = EpisodeDataset(alex_exports[A1_JOINT_DELTA])
    assert a1.action_dim == FakeDoorPushEnv.N_JOINTS
    assert validate_dataset(a1).ok


def test_episode_ids_shared_across_action_spaces(alex_exports) -> None:
    ids = {
        space: sorted(EpisodeDataset(path).episode_ids)
        for space, path in alex_exports.items()
        if space != A4_OBJ_CENTRIC_CHUNK
    }
    a4_ids = sorted(A4ChunkDataset(alex_exports[A4_OBJ_CENTRIC_CHUNK]).episode_ids)
    reference = ids[A2_EE_DELTA]
    assert all(episode_ids == reference for episode_ids in ids.values())
    assert a4_ids == reference


def test_core_preset_is_9dim_everywhere(synthetic_a2, alex_a2) -> None:
    for dataset in (synthetic_a2, alex_a2):
        obs = obs_matrix(dataset[0], "core")
        assert obs.shape == (dataset[0].n_steps, 9)
        assert np.isfinite(obs).all()


def test_core_contact_uses_sensed_when_available(synthetic_a2, alex_a2) -> None:
    assert obs_matrix(synthetic_a2[0], "core_contact").shape[1] == 10
    alex_obs = obs_matrix(alex_a2[0], "core_contact")
    assert alex_obs.shape[1] == 10
    np.testing.assert_array_equal(alex_obs[:, -1], alex_a2[0].obs["sensed"])


def test_unknown_obs_preset_is_rejected(synthetic_a2) -> None:
    with pytest.raises(ValueError, match="unknown obs preset"):
        obs_matrix(synthetic_a2[0], "nope")


def test_core_door_pose_preset_is_14dim_and_encodes_yaw(tmp_path) -> None:
    yaw = 0.6
    origin = (1.0, -2.0, 0.5)
    episode = run_episode(
        FakeDoorPushEnv(yaw_rad=yaw, origin=origin),
        plan_episodes(1, 0, 0)[0],
        make_test_engine_cfg(),
    )
    exported = export_datasets([episode], tmp_path, version="v0")
    dataset = EpisodeDataset(exported[A2_EE_DELTA])
    obs = obs_matrix(dataset[0], "core_door_pose")
    assert obs.shape == (dataset[0].n_steps, 14)
    assert np.isfinite(obs).all()
    # First 9 dims identical to core; door-pose block is constant per episode.
    np.testing.assert_array_equal(obs[:, :9], obs_matrix(dataset[0], "core"))
    np.testing.assert_allclose(obs[:, 9:12], np.tile(origin, (obs.shape[0], 1)), atol=1e-12)
    np.testing.assert_allclose(obs[:, 12], np.sin(yaw), atol=1e-12)
    np.testing.assert_allclose(obs[:, 13], np.cos(yaw), atol=1e-12)


def test_core_door_pose_preset_fails_clearly_on_old_episodes(alex_a2) -> None:
    """Episodes recorded before the door-pose terms existed must be rejected."""
    import dataclasses

    record = alex_a2[0]
    stripped_obs = {
        key: value
        for key, value in record.obs.items()
        if not key.startswith("door_rel_pos") and key != "door_yaw_rad"
    }
    old_record = dataclasses.replace(record, obs=stripped_obs)
    with pytest.raises(ValueError, match="core_door_pose"):
        obs_matrix(old_record, "core_door_pose")


def test_a4_dataset_parses_and_validates_chunks(alex_exports) -> None:
    a4 = A4ChunkDataset(alex_exports[A4_OBJ_CENTRIC_CHUNK])
    assert len(a4) == N_EPISODES
    record = a4[0]
    assert record.chunks and all(c.phase in A4_PHASE_VOCAB for c in record.chunks)
    assert validate_a4_dataset(a4).ok


def test_a4_validation_fails_closed_on_missing_outcome(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_missing_outcome")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["outcome"] = None
    _write_jsonl_records(jsonl, records)

    with pytest.raises(ValueError, match="outcome"):
        A4ChunkDataset(a4_dir)


def test_a4_validation_rejects_duplicate_ids(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_duplicate")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[1]["meta"]["episode_id"] = records[0]["meta"]["episode_id"]
    _write_jsonl_records(jsonl, records)

    result = validate_a4_dataset(A4ChunkDataset(a4_dir))

    assert not result.ok
    assert any("duplicate episode ids" in error for error in result.errors)


def test_a4_validation_rejects_bad_target_width(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_bad_target")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["chunks"][0]["contact_target_panel"] = [0.1, 0.2]
    _write_jsonl_records(jsonl, records)

    result = validate_a4_dataset(A4ChunkDataset(a4_dir))

    assert not result.ok
    assert any("contact_target_panel" in error for error in result.errors)


def test_validate_passes_on_exported_datasets(synthetic_a2, alex_a2, alex_exports) -> None:
    assert validate_dataset(synthetic_a2, A2_EE_DELTA).ok
    assert validate_dataset(alex_a2, A2_EE_DELTA).ok
    assert validate_dataset(EpisodeDataset(alex_exports[A3_OBJ_REL_EE_DELTA])).ok


def test_validate_catches_planted_defects(synthetic_a2) -> None:
    record = synthetic_a2[0]

    mismatch = validate_episode(record, expected_space=A3_OBJ_REL_EE_DELTA)
    assert any("does not match" in e for e in mismatch.errors)

    bad_actions = record.actions.copy()
    bad_actions[3, 0] = np.nan
    non_finite = validate_episode(dataclasses.replace(record, actions=bad_actions))
    assert any("non-finite action" in e for e in non_finite.errors)

    unknown = validate_episode(dataclasses.replace(record, schema_version="phase9.v9"))
    assert any("unknown schema_version" in e for e in unknown.errors)

    truncated = validate_episode(dataclasses.replace(record, t=record.t[:-2]))
    assert any("inconsistent step counts" in e for e in truncated.errors)


def test_validate_rejects_unknown_termination_reason(synthetic_a2) -> None:
    record = synthetic_a2[0]
    result = validate_episode(
        dataclasses.replace(record, success=False, termination_reason="novel_interpretation")
    )

    assert any("unknown termination_reason" in error for error in result.errors)


def test_validate_rejects_bad_timing_and_control_dt(synthetic_a2) -> None:
    record = synthetic_a2[0]
    bad_t = record.t.copy()
    bad_t[1] += record.meta["control_dt"] * 0.5
    result = validate_episode(dataclasses.replace(record, t=bad_t))
    assert any("timestamp deltas" in error for error in result.errors)

    bad_meta = dict(record.meta)
    bad_meta["control_dt"] = -0.01
    result = validate_episode(dataclasses.replace(record, meta=bad_meta))
    assert any("control_dt" in error for error in result.errors)


def test_validate_rejects_bad_contact_flags_and_sources(synthetic_a2) -> None:
    record = synthetic_a2[0]
    bad_obs = dict(record.obs)
    bad_obs["inferred"] = bad_obs["inferred"].copy()
    bad_obs["inferred"][0] = 0.5
    result = validate_episode(dataclasses.replace(record, obs=bad_obs))
    assert any("contact flag" in error for error in result.errors)

    bad_contact = dict(record.buffer.steps[0].contact)
    bad_contact["source"] = "mystery_sensor"
    bad_step = dataclasses.replace(record.buffer.steps[0], contact=bad_contact)
    bad_buffer = dataclasses.replace(record.buffer, steps=[bad_step, *record.buffer.steps[1:]])
    result = validate_episode(dataclasses.replace(record, buffer=bad_buffer))
    assert any("contact source" in error for error in result.errors)


def test_validate_episode_reports_malformed_observation_shape(synthetic_a2) -> None:
    record = synthetic_a2[0]
    bad_obs = dict(record.obs)
    bad_obs["door_angle_rad"] = np.asarray(0.0)

    result = validate_episode(dataclasses.replace(record, obs=bad_obs))

    assert not result.ok
    assert any("obs 'door_angle_rad'" in error and "shape" in error for error in result.errors)


def test_validate_rejects_mislabeled_a3_actions(alex_exports) -> None:
    a3 = EpisodeDataset(alex_exports[A3_OBJ_REL_EE_DELTA])
    record = a3[0]
    bad_actions = record.actions.copy()
    bad_actions[0, 0] += 0.25

    result = validate_episode(dataclasses.replace(record, actions=bad_actions))

    assert any("action_door_frame" in error for error in result.errors)


def test_validate_dataset_reports_malformed_meta_and_action_rank(
    synthetic_exports, tmp_path
) -> None:
    dataset_dir = _copy_dataset(synthetic_exports[A2_EE_DELTA], tmp_path, "bad_meta")
    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["action_space"]
    meta_path.write_text(json.dumps(meta) + "\n")

    result = validate_dataset(EpisodeDataset(dataset_dir))
    assert not result.ok
    assert any("action_space" in error for error in result.errors)

    rank_dir = _copy_dataset(synthetic_exports[A2_EE_DELTA], tmp_path, "bad_action_rank")
    dataset = EpisodeDataset(rank_dir)
    n_steps = dataset[0].n_steps
    import h5py

    h5_path = sorted(rank_dir.glob("episode_*.hdf5"))[0]
    with h5py.File(h5_path, "r+") as h5:
        del h5["steps/action"]
        h5["steps"].create_dataset("action", data=np.zeros(n_steps))

    result = validate_dataset(EpisodeDataset(rank_dir))
    assert not result.ok
    assert any("rank 2" in error for error in result.errors)


def test_matched_action_space_validation_rejects_same_id_mismatched_content(alex_exports) -> None:
    a2 = EpisodeDataset(alex_exports[A2_EE_DELTA])
    a3 = EpisodeDataset(alex_exports[A3_OBJ_REL_EE_DELTA])
    a3.records = list(a3.records)
    bad_meta = dict(a3[0].meta)
    bad_meta["seed"] = int(bad_meta["seed"]) + 1000
    bad_obs = dict(a3[0].obs)
    bad_obs["door_angle_rad"] = bad_obs["door_angle_rad"].copy()
    bad_obs["door_angle_rad"][0] += 1.0
    a3.records[0] = dataclasses.replace(a3[0], meta=bad_meta, obs=bad_obs)

    result = validate_matched_action_space_datasets({A2_EE_DELTA: a2, A3_OBJ_REL_EE_DELTA: a3})

    assert not result.ok
    assert any("meta.seed differs" in error for error in result.errors)
    assert any("core low-dim observations differ" in error for error in result.errors)


def test_norm_stats_roundtrip_and_positive_std(synthetic_a2, tmp_path) -> None:
    train_ids = synthetic_a2.episode_ids[:3]
    stats = compute_norm_stats(synthetic_a2, train_ids)
    assert stats.action.dim == EE_DELTA_DIM
    assert stats.obs.dim == 9 and stats.obs_preset == "core"
    assert (stats.action.std > 0.0).all()

    actions = synthetic_a2[0].actions
    np.testing.assert_allclose(
        stats.action.denormalize(stats.action.normalize(actions)), actions, atol=1e-9
    )

    path = save_norm_stats(norm_stats_path(tmp_path), stats)
    loaded = load_norm_stats(path)
    np.testing.assert_array_equal(loaded.action.mean, stats.action.mean)
    np.testing.assert_array_equal(loaded.obs.std, stats.obs.std)
    assert loaded.train_episode_ids == tuple(train_ids)
    assert validate_norm_stats(loaded, synthetic_a2, train_ids) == []

    pose_stats = compute_norm_stats(synthetic_a2, train_ids, obs_preset="core_door_pose")
    assert pose_stats.obs_preset == "core_door_pose"
    assert pose_stats.obs.dim > stats.obs.dim


def test_norm_stats_validation_rejects_stale_or_wrong_dimension_stats(synthetic_a2) -> None:
    train_ids = synthetic_a2.episode_ids[:3]
    stats = compute_norm_stats(synthetic_a2, train_ids)

    stale_action = dataclasses.replace(
        stats.action,
        mean=stats.action.mean + 1.0,
    )
    stale = dataclasses.replace(stats, action=stale_action)
    assert any(
        "recomputed action mean" in error
        for error in validate_norm_stats(stale, synthetic_a2, train_ids)
    )

    wrong_train = validate_norm_stats(stats, synthetic_a2, list(reversed(train_ids)))
    assert any("train_episode_ids" in error for error in wrong_train)

    bad_action = dataclasses.replace(
        stats.action,
        mean=stats.action.mean[:-1],
        std=stats.action.std[:-1],
        min=stats.action.min[:-1],
        max=stats.action.max[:-1],
    )
    wrong_dim = dataclasses.replace(stats, action=bad_action)
    assert any(
        "action dim" in error for error in validate_norm_stats(wrong_dim, synthetic_a2, train_ids)
    )


def test_verify_dataset_interface_default_does_not_rewrite_artifacts(alex_exports) -> None:
    verify = _load_script("scripts/verify_dataset_interface.py")
    root = alex_exports[A2_EE_DELTA].parents[2]
    args = argparse.Namespace(
        datasets_root=root,
        version="v0",
        horizon=20,
        batch_size=8,
        seed=0,
        write_artifacts=True,
    )
    assert verify.verify_task(args, "door_push") == []

    artifact_paths = [splits_path(root, "door_push", "v0")]
    artifact_paths.extend(
        norm_stats_path(path)
        for space, path in alex_exports.items()
        if space != A4_OBJ_CENTRIC_CHUNK
    )
    before = {path: path.read_bytes() for path in artifact_paths}
    args.write_artifacts = False

    failures = verify.verify_task(args, "door_push")

    assert failures == []
    assert {path: path.read_bytes() for path in artifact_paths} == before


def test_verify_dataset_interface_requires_a4_by_default(alex_exports, tmp_path) -> None:
    verify = _load_script("scripts/verify_dataset_interface.py")
    source_root = alex_exports[A2_EE_DELTA].parents[2]
    root = tmp_path / "datasets"
    shutil.copytree(source_root, root)
    shutil.rmtree(root / "door_push" / A4_OBJ_CENTRIC_CHUNK)
    args = argparse.Namespace(
        datasets_root=root,
        version="v0",
        horizon=20,
        batch_size=8,
        seed=0,
        write_artifacts=False,
    )

    failures = verify.verify_task(args, "door_push")

    assert any(
        "missing required" in failure and A4_OBJ_CENTRIC_CHUNK in failure for failure in failures
    )


def test_chunk_sampler_windows_and_pads(synthetic_a2) -> None:
    horizon = 10
    sampler = ChunkSampler(synthetic_a2, horizon=horizon)
    assert len(sampler) == sum(r.n_steps for r in synthetic_a2.records)
    assert sampler.obs_dim == 9 and sampler.action_dim == EE_DELTA_DIM

    record = synthetic_a2[0]
    mid = sampler.sample(5)
    np.testing.assert_array_equal(mid.actions, record.actions[5 : 5 + horizon])
    assert not mid.is_pad.any()
    np.testing.assert_array_equal(mid.obs, obs_matrix(record, "core")[5])

    last = sampler.sample(record.n_steps - 1)
    np.testing.assert_array_equal(last.actions[0], record.actions[-1])
    assert not last.is_pad[0] and last.is_pad[1:].all()
    assert (last.actions[1:] == 0.0).all()


def test_chunk_sampler_respects_split_restriction(synthetic_a2) -> None:
    chosen = [synthetic_a2.episode_ids[1]]
    sampler = ChunkSampler(synthetic_a2, horizon=4, episode_ids=chosen)
    assert len(sampler) == sum(synthetic_a2.by_id(e).n_steps for e in chosen)
    np.testing.assert_array_equal(
        sampler.sample(0).actions[0], synthetic_a2.by_id(chosen[0]).actions[0]
    )


def test_batch_iterator_is_seeded_and_shaped(synthetic_a2) -> None:
    sampler = ChunkSampler(synthetic_a2, horizon=8)
    iterator = BatchIterator(sampler, batch_size=16, seed=3)
    first_pass = list(iterator)
    second_pass = list(BatchIterator(sampler, batch_size=16, seed=3))
    assert len(first_pass) == len(iterator)
    for a, b in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(a["obs"], b["obs"])
        np.testing.assert_array_equal(a["actions"], b["actions"])
        np.testing.assert_array_equal(a["is_pad"], b["is_pad"])

    batch = first_pass[0]
    assert set(batch) == {"obs", "actions", "is_pad"}
    assert batch["obs"].shape == (16, 9)
    assert batch["actions"].shape == (16, 8, EE_DELTA_DIM)
    assert batch["is_pad"].shape == (16, 8) and batch["is_pad"].dtype == bool
    dropped = list(BatchIterator(sampler, batch_size=100, seed=0, drop_last=True))
    assert all(b["obs"].shape[0] == 100 for b in dropped)
