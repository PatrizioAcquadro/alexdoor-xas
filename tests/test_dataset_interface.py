"""Pure tests for the Phase 3.0 dataset/model interface (loader -> batches)."""

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
    EE_DELTA_DIM,
)
from alexdoor_xas.data_engine import DataEngineCfg, export_datasets, plan_episodes, run_episode
from alexdoor_xas.dataset import (
    A4_FEATURE_DIM,
    A4_PHASE_VOCAB,
    STD_FLOOR,
    A4ChunkDataset,
    BatchIterator,
    ChunkSampler,
    EpisodeDataset,
    compute_norm_stats,
    episode_chunk_features,
    load_norm_stats,
    load_splits,
    make_splits,
    norm_stats_path,
    obs_matrix,
    save_norm_stats,
    save_splits,
    splits_path,
    validate_dataset,
    validate_dataset_dir,
    validate_episode,
    validate_matched_action_space_datasets,
    validate_norm_stats,
)
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv

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
            DataEngineCfg(),
        )
        for seed in range(N_EPISODES)
    ]
    return export_datasets(episodes, tmp_root, version="v0")


@pytest.fixture(scope="module")
def proxy_exports(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("proxy"), FakeDoorPushEnv)


@pytest.fixture(scope="module")
def alex_exports(tmp_path_factory):
    return _export(tmp_path_factory.mktemp("alex"), FakeForceDoorPushEnv)


@pytest.fixture(scope="module")
def proxy_a2(proxy_exports) -> EpisodeDataset:
    return EpisodeDataset(proxy_exports[A2_EE_DELTA])


@pytest.fixture(scope="module")
def alex_a2(alex_exports) -> EpisodeDataset:
    return EpisodeDataset(alex_exports[A2_EE_DELTA])


# ── loading ───────────────────────────────────────────────────────────────────


def test_dataset_loads_records_with_stacked_arrays(proxy_a2) -> None:
    assert len(proxy_a2) == N_EPISODES
    assert proxy_a2.action_space == A2_EE_DELTA
    assert proxy_a2.action_dim == EE_DELTA_DIM
    record = proxy_a2[0]
    assert record.n_steps > 0
    assert record.actions.shape == (record.n_steps, EE_DELTA_DIM)
    assert record.t.shape == (record.n_steps,)
    assert record.schema_version == "phase2.v1"
    assert record.success and record.failure_label is None
    assert {"ee_pos_w", "ee_quat_w_xyzw", "door_angle_rad", "inferred"} <= set(record.obs)
    assert proxy_a2.by_id(record.episode_id) is record
    with pytest.raises(KeyError):
        proxy_a2.by_id("no-such-episode")


def test_a1_dataset_has_joint_wide_actions(alex_exports) -> None:
    a1 = EpisodeDataset(alex_exports[A1_JOINT_DELTA])
    assert a1.action_dim == FakeForceDoorPushEnv.N_JOINTS
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


# ── observation presets ───────────────────────────────────────────────────────


def test_core_preset_is_9dim_everywhere(proxy_a2, alex_a2) -> None:
    for dataset in (proxy_a2, alex_a2):
        obs = dataset.obs(0, "core")
        assert obs.shape == (dataset[0].n_steps, 9)
        assert np.isfinite(obs).all()


def test_core_contact_uses_sensed_when_available(proxy_a2, alex_a2) -> None:
    assert proxy_a2.obs(0, "core_contact").shape[1] == 10
    alex_obs = alex_a2.obs(0, "core_contact")
    assert alex_obs.shape[1] == 10
    np.testing.assert_array_equal(alex_obs[:, -1], alex_a2[0].obs["sensed"])


def test_alex_full_preset_requires_force_sensing_episodes(proxy_a2, alex_a2) -> None:
    n_joints = FakeForceDoorPushEnv.N_JOINTS
    assert alex_a2.obs(0, "alex_full").shape[1] == 2 * n_joints + 11
    with pytest.raises(ValueError, match="alex_full"):
        proxy_a2.obs(0, "alex_full")
    with pytest.raises(ValueError, match="unknown obs preset"):
        proxy_a2.obs(0, "nope")


def test_frozen_presets_stay_bit_compatible() -> None:
    """The Phase 3.0 preset freeze: key tuples must never change (additive-only)."""
    from alexdoor_xas.dataset import OBS_PRESETS

    assert OBS_PRESETS["core"] == (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
    )
    assert OBS_PRESETS["core_contact"] == OBS_PRESETS["core"] + ("contact_flag",)
    assert OBS_PRESETS["alex_full"] == OBS_PRESETS["core"] + (
        "joint_pos",
        "joint_vel",
        "force_n",
        "sensed",
    )


def test_core_door_pose_preset_is_14dim_and_encodes_yaw(tmp_path) -> None:
    yaw = 0.6
    origin = (1.0, -2.0, 0.5)
    episode = run_episode(
        FakeDoorPushEnv(yaw_rad=yaw, origin=origin),
        plan_episodes(1, 0, 0)[0],
        DataEngineCfg(),
    )
    exported = export_datasets([episode], tmp_path, version="v0")
    dataset = EpisodeDataset(exported[A2_EE_DELTA])
    obs = dataset.obs(0, "core_door_pose")
    assert obs.shape == (dataset[0].n_steps, 14)
    assert np.isfinite(obs).all()
    # First 9 dims identical to core; door-pose block is constant per episode.
    np.testing.assert_array_equal(obs[:, :9], dataset.obs(0, "core"))
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


# ── A4 ────────────────────────────────────────────────────────────────────────


def test_a4_dataset_parses_chunks_and_encodes_features(alex_exports) -> None:
    a4 = A4ChunkDataset(alex_exports[A4_OBJ_CENTRIC_CHUNK])
    assert len(a4) == N_EPISODES
    record = a4[0]
    assert record.chunks and all(c.phase in A4_PHASE_VOCAB for c in record.chunks)
    features = episode_chunk_features(record)
    assert features.shape == (len(record.chunks), A4_FEATURE_DIM)
    assert np.isfinite(features).all()
    # One-hot block sums to 1 per chunk; duration column is positive seconds.
    np.testing.assert_array_equal(features[:, : len(A4_PHASE_VOCAB)].sum(axis=1), 1.0)
    assert (features[:, -1] > 0).all()


def test_a4_phase_vocab_pins_the_scripted_fsm() -> None:
    from alexdoor_xas.policies.scripted import DoorPushPhase

    emitting = tuple(p.value for p in DoorPushPhase if p is not DoorPushPhase.DONE)
    assert A4_PHASE_VOCAB == emitting


def test_a4_validation_fails_closed_on_missing_outcome(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_missing_outcome")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["outcome"] = None
    _write_jsonl_records(jsonl, records)

    result = validate_dataset_dir(a4_dir)

    assert not result.ok
    assert any("outcome" in error for error in result.errors)


def test_a4_validation_rejects_duplicate_ids(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_duplicate")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[1]["meta"]["episode_id"] = records[0]["meta"]["episode_id"]
    _write_jsonl_records(jsonl, records)

    result = validate_dataset_dir(a4_dir)

    assert not result.ok
    assert any("duplicate A4 episode ids" in error for error in result.errors)


def test_a4_validation_rejects_bad_target_width(alex_exports, tmp_path) -> None:
    a4_dir = _copy_dataset(alex_exports[A4_OBJ_CENTRIC_CHUNK], tmp_path, "a4_bad_target")
    jsonl = a4_dir / "episodes.jsonl"
    records = _jsonl_records(jsonl)
    records[0]["chunks"][0]["contact_target_panel"] = [0.1, 0.2]
    _write_jsonl_records(jsonl, records)

    result = validate_dataset_dir(a4_dir)

    assert not result.ok
    assert any("contact_target_panel" in error for error in result.errors)
    assert any("A4 feature encoding" in error for error in result.errors)


# ── validation ────────────────────────────────────────────────────────────────


def test_validate_passes_on_exported_datasets(proxy_a2, alex_a2, alex_exports) -> None:
    assert validate_dataset(proxy_a2, A2_EE_DELTA).ok
    assert validate_dataset(alex_a2, A2_EE_DELTA).ok
    assert validate_dataset(EpisodeDataset(alex_exports[A3_OBJ_REL_EE_DELTA])).ok


def test_validate_catches_planted_defects(proxy_a2) -> None:
    record = proxy_a2[0]

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


def test_validate_rejects_unknown_failure_labels(proxy_a2) -> None:
    record = proxy_a2[0]
    result = validate_episode(
        dataclasses.replace(record, success=False, failure_label="novel_failure_mode")
    )

    assert any("not in the frozen vocabulary" in error for error in result.errors)


def test_validate_rejects_bad_timing_and_control_dt(proxy_a2) -> None:
    record = proxy_a2[0]
    bad_t = record.t.copy()
    bad_t[1] += record.meta["control_dt"] * 0.5
    result = validate_episode(dataclasses.replace(record, t=bad_t))
    assert any("timestamp deltas" in error for error in result.errors)

    bad_meta = dict(record.meta)
    bad_meta["control_dt"] = -0.01
    result = validate_episode(dataclasses.replace(record, meta=bad_meta))
    assert any("control_dt" in error for error in result.errors)


def test_validate_rejects_bad_contact_flags_and_sources(proxy_a2) -> None:
    record = proxy_a2[0]
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


def test_validate_episode_reports_malformed_observation_shape(proxy_a2) -> None:
    record = proxy_a2[0]
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


def test_validate_dataset_dir_reports_malformed_meta_and_action_rank(
    proxy_exports, tmp_path
) -> None:
    dataset_dir = _copy_dataset(proxy_exports[A2_EE_DELTA], tmp_path, "bad_meta")
    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["action_space"]
    meta_path.write_text(json.dumps(meta) + "\n")

    result = validate_dataset_dir(dataset_dir)
    assert not result.ok
    assert any("action_space" in error for error in result.errors)

    rank_dir = _copy_dataset(proxy_exports[A2_EE_DELTA], tmp_path, "bad_action_rank")
    n_steps = EpisodeDataset(rank_dir)[0].n_steps
    import h5py

    h5_path = next(rank_dir.glob("episode_*.hdf5"))
    with h5py.File(h5_path, "r+") as h5:
        del h5["steps/action"]
        h5["steps"].create_dataset("action", data=np.zeros(n_steps))

    result = validate_dataset_dir(rank_dir)
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


# ── splits ────────────────────────────────────────────────────────────────────


def test_splits_are_deterministic_disjoint_and_covering(proxy_a2, tmp_path) -> None:
    ids = proxy_a2.episode_ids
    splits = make_splits(ids, seed=7)
    assert make_splits(ids, seed=7) == splits
    assert make_splits(ids, seed=8) != splits
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_ids) == sorted(ids)
    assert len(set(all_ids)) == len(ids)
    assert len(splits["val"]) >= 1 and len(splits["test"]) >= 1 and len(splits["train"]) >= 1

    path = save_splits(tmp_path / "splits" / "v0.json", splits, seed=7)
    assert load_splits(path, episode_ids=ids) == splits
    with pytest.raises(ValueError, match="re-exported"):
        load_splits(path, episode_ids=["stale-1", "stale-2", "stale-3", "stale-4"])
    with pytest.raises(ValueError, match="at least 3"):
        make_splits(ids[:2])


# ── normalization ─────────────────────────────────────────────────────────────


def test_norm_stats_roundtrip_and_zero_std_guard(proxy_a2, tmp_path) -> None:
    train_ids = proxy_a2.episode_ids[:3]
    stats = compute_norm_stats(proxy_a2, train_ids)
    assert stats.action.dim == EE_DELTA_DIM
    assert stats.obs.dim == 9 and stats.obs_preset == "core"
    # The proxy never actuates rotations: those dims are constant zero -> floored.
    assert (stats.action.std[3:] == STD_FLOOR).all()
    assert (stats.action.std[:3] > STD_FLOOR).any()

    actions = proxy_a2[0].actions
    np.testing.assert_allclose(
        stats.action.denormalize(stats.action.normalize(actions)), actions, atol=1e-9
    )

    path = save_norm_stats(norm_stats_path(tmp_path), stats)
    loaded = load_norm_stats(path)
    np.testing.assert_array_equal(loaded.action.mean, stats.action.mean)
    np.testing.assert_array_equal(loaded.obs.std, stats.obs.std)
    assert loaded.train_episode_ids == tuple(train_ids)
    assert validate_norm_stats(loaded, proxy_a2, train_ids) == []


def test_norm_stats_validation_rejects_stale_or_wrong_dimension_stats(proxy_a2) -> None:
    train_ids = proxy_a2.episode_ids[:3]
    stats = compute_norm_stats(proxy_a2, train_ids)

    stale = dataclasses.replace(stats, dataset_fingerprint="stale")
    assert any("fingerprint" in error for error in validate_norm_stats(stale, proxy_a2, train_ids))

    wrong_train = validate_norm_stats(stats, proxy_a2, list(reversed(train_ids)))
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
        "action dim" in error for error in validate_norm_stats(wrong_dim, proxy_a2, train_ids)
    )


# ── scripts ───────────────────────────────────────────────────────────────────


def test_verify_dataset_interface_default_does_not_rewrite_artifacts(
    proxy_exports, tmp_path
) -> None:
    verify = _load_script("scripts/verify_dataset_interface.py")
    root = proxy_exports[A2_EE_DELTA].parents[2]
    split_file = splits_path(root, "door_push", "v0")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text("sentinel split\n")
    stats_file = norm_stats_path(proxy_exports[A2_EE_DELTA])
    stats_file.write_text("sentinel stats\n")

    args = argparse.Namespace(
        datasets_root=root,
        version="v0",
        horizon=20,
        batch_size=8,
        seed=0,
        write_artifacts=False,
        allow_missing_a4=False,
        artifacts_root=tmp_path / "gate_tmp",
    )

    failures = verify.verify_task(args, "door_push")

    assert failures == []
    assert split_file.read_text() == "sentinel split\n"
    assert stats_file.read_text() == "sentinel stats\n"


def test_verify_dataset_interface_requires_a4_by_default(proxy_exports, tmp_path) -> None:
    verify = _load_script("scripts/verify_dataset_interface.py")
    source_root = proxy_exports[A2_EE_DELTA].parents[2]
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
        allow_missing_a4=False,
        artifacts_root=tmp_path / "gate_tmp",
    )

    failures = verify.verify_task(args, "door_push")

    assert any(
        "missing required" in failure and A4_OBJ_CENTRIC_CHUNK in failure
        for failure in failures
    )


def test_inspect_dataset_uses_split_episode_and_masks_padded_actions(proxy_a2, tmp_path) -> None:
    inspect = _load_script("scripts/inspect_dataset.py")
    ids = proxy_a2.episode_ids
    splits = {"train": [ids[1], ids[2]], "val": [ids[3]], "test": [ids[0]]}
    save_splits(splits_path(proxy_a2.dataset_dir.parents[2], proxy_a2.task, "v0"), splits)
    args = argparse.Namespace(dataset=proxy_a2.dataset_dir, split="train")

    selected = inspect._selected_episode_ids(args, proxy_a2.episode_ids)
    record = inspect._plot_record(proxy_a2, selected)

    assert selected == splits["train"]
    assert record.episode_id == ids[1]

    sampler = ChunkSampler(proxy_a2, horizon=proxy_a2[0].n_steps + 5, episode_ids=[ids[0]])
    sample = sampler.sample(proxy_a2[0].n_steps - 1)
    batch = {
        "actions": sample.actions.reshape(1, *sample.actions.shape),
        "is_pad": sample.is_pad.reshape(1, *sample.is_pad.shape),
    }
    valid = inspect._valid_batch_actions(batch)
    assert valid.shape == (1, proxy_a2.action_dim)
    np.testing.assert_array_equal(valid[0], proxy_a2[0].actions[-1])


# ── chunk sampling + batching ─────────────────────────────────────────────────


def test_chunk_sampler_windows_and_pads(proxy_a2) -> None:
    horizon = 10
    sampler = ChunkSampler(proxy_a2, horizon=horizon)
    assert len(sampler) == sum(r.n_steps for r in proxy_a2.records)
    assert sampler.obs_dim == 9 and sampler.action_dim == EE_DELTA_DIM

    record = proxy_a2[0]
    mid = sampler.sample(5)
    assert mid.episode_id == record.episode_id and mid.t_index == 5
    np.testing.assert_array_equal(mid.actions, record.actions[5 : 5 + horizon])
    assert not mid.is_pad.any()
    np.testing.assert_array_equal(mid.obs, obs_matrix(record, "core")[5])

    last = sampler.sample(record.n_steps - 1)  # 1 valid action, H-1 padded
    assert last.t_index == record.n_steps - 1
    np.testing.assert_array_equal(last.actions[0], record.actions[-1])
    assert not last.is_pad[0] and last.is_pad[1:].all()
    assert (last.actions[1:] == 0.0).all()


def test_chunk_sampler_respects_split_restriction(proxy_a2) -> None:
    chosen = proxy_a2.episode_ids[:2]
    sampler = ChunkSampler(proxy_a2, horizon=4, episode_ids=chosen)
    assert len(sampler) == sum(proxy_a2.by_id(e).n_steps for e in chosen)
    seen = {sampler.sample(i).episode_id for i in range(len(sampler))}
    assert seen == set(chosen)


def test_batch_iterator_is_seeded_and_shaped(proxy_a2) -> None:
    sampler = ChunkSampler(proxy_a2, horizon=8)
    iterator = BatchIterator(sampler, batch_size=16, seed=3)
    first_pass = list(iterator)
    second_pass = list(BatchIterator(sampler, batch_size=16, seed=3))
    assert len(first_pass) == len(iterator)
    for a, b in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(a["obs"], b["obs"])
        np.testing.assert_array_equal(a["actions"], b["actions"])
        np.testing.assert_array_equal(a["t"], b["t"])
        np.testing.assert_array_equal(a["t_index"], b["t_index"])
        np.testing.assert_array_equal(a["t_s"], b["t_s"])
        assert a["episode_ids"] == b["episode_ids"]

    batch = first_pass[0]
    assert batch["obs"].shape == (16, 9)
    assert batch["actions"].shape == (16, 8, EE_DELTA_DIM)
    assert batch["is_pad"].shape == (16, 8) and batch["is_pad"].dtype == bool
    np.testing.assert_array_equal(batch["t"], batch["t_index"])
    assert batch["t_s"].shape == (16,)
    assert batch["action_space"] == A2_EE_DELTA

    dropped = list(BatchIterator(sampler, batch_size=100, seed=0, drop_last=True))
    assert all(b["obs"].shape[0] == 100 for b in dropped)


def test_dummy_model_consumes_a_batch(proxy_a2) -> None:
    """The Phase 3.0 acceptance shape-check, without any policy code."""
    horizon, batch_size = 8, 16
    splits = make_splits(proxy_a2.episode_ids, seed=0)
    stats = compute_norm_stats(proxy_a2, splits["train"])
    sampler = ChunkSampler(proxy_a2, horizon=horizon, episode_ids=splits["train"])
    batch = next(iter(BatchIterator(sampler, batch_size=batch_size, seed=0)))

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((sampler.obs_dim, horizon * sampler.action_dim))
    predicted = (stats.obs.normalize(batch["obs"]) @ weights).reshape(
        batch_size, horizon, sampler.action_dim
    )
    assert predicted.shape == batch["actions"].shape
    error = predicted - stats.action.normalize(batch["actions"])
    loss = float((error**2 * ~batch["is_pad"][..., None]).mean())
    assert np.isfinite(loss)
