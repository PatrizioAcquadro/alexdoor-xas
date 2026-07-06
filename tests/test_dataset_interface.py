"""Pure tests for the Phase 3.0 dataset/model interface (loader -> batches)."""

from __future__ import annotations

import dataclasses
import importlib.util

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
    validate_dataset,
    validate_episode,
)
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)
pytestmark = requires_h5py

N_EPISODES = 4


def _export(tmp_root, env_factory):
    episodes = [
        run_episode(env_factory(), plan_episodes(1, 0, seed)[0], DataEngineCfg())
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
        assert a["episode_ids"] == b["episode_ids"]

    batch = first_pass[0]
    assert batch["obs"].shape == (16, 9)
    assert batch["actions"].shape == (16, 8, EE_DELTA_DIM)
    assert batch["is_pad"].shape == (16, 8) and batch["is_pad"].dtype == bool
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
