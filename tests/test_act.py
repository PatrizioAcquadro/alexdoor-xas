"""Pure tests for the ACT baseline package (Phase 3.2). No Isaac imports."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from alexdoor_xas.adapters.a2 import A2Adapter
from alexdoor_xas.adapters.rollout import read_door_frame, read_step_context, rollout_chunks
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.dataset.loader import EpisodeRecord
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.policies.act.config import ActModelCfg, ActTrainCfg
from alexdoor_xas.policies.act.model import ACTModel, act_loss
from alexdoor_xas.policies.act.policy import ActPolicy, act_chunk_source
from alexdoor_xas.policies.act.train import make_seeded_model, train_act
from alexdoor_xas.policies.common.checkpoint import (
    ACT_CHECKPOINT_FORMAT,
    save_checkpoint_payload,
)
from alexdoor_xas.policies.common.inspect import open_loop_report
from alexdoor_xas.policies.common.obs import build_rollout_obs, read_door_pose_obs
from conftest import (
    TEST_ROBOT_LIMITS,
    FakeDoorPushEnv,
    make_test_engine_cfg,
)

TINY_MODEL_CFG = ActModelCfg(
    chunk_size=8,
    d_model=32,
    n_heads=2,
    dim_feedforward=64,
    z_dim=4,
    cvae_encoder_layers=1,
    encoder_layers=1,
    decoder_layers=1,
    dropout=0.0,
)
OBS_DIM = 9
ACTION_DIM = 6
TEST_ROBOT_ASSET = RobotAssetRef("alex_v2_test", "a" * 64)


def _tiny_model(seed: int = 0) -> ACTModel:
    torch.manual_seed(seed)
    return ACTModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, cfg=TINY_MODEL_CFG)


def _tiny_batch(batch: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    horizon = TINY_MODEL_CFG.chunk_size
    is_pad = torch.zeros(batch, horizon, dtype=torch.bool)
    is_pad[:, horizon - 2 :] = True
    return {
        "obs": torch.randn(batch, OBS_DIM, generator=generator),
        "actions": torch.randn(batch, horizon, ACTION_DIM, generator=generator),
        "is_pad": is_pad,
    }


def _tiny_stats() -> DatasetNormStats:
    rows_a = [np.arange(12, dtype=np.float64).reshape(2, ACTION_DIM) * 0.01]
    rows_o = [np.arange(18, dtype=np.float64).reshape(2, OBS_DIM) * 0.1]
    return DatasetNormStats(
        action=NormStats.from_rows(rows_a),
        obs=NormStats.from_rows(rows_o),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )


def _checkpoint_config(obs_preset: str = "core") -> dict:
    return {
        "dataset": {
            "task": "door_push_alex_v2",
            "space": "A2_ee_delta",
            "version": "v2_pose",
            "view_id": None,
            "obs_preset": obs_preset,
        }
    }


def _save_checkpoint(path, model, config, stats, robot_asset=TEST_ROBOT_ASSET):
    return save_checkpoint_payload(
        path,
        ACT_CHECKPOINT_FORMAT,
        model,
        config,
        stats,
        {},
        robot_asset,
    )


def test_forward_shapes_and_finite() -> None:
    model = _tiny_model()
    batch = _tiny_batch()
    a_hat, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])

    assert a_hat.shape == (4, TINY_MODEL_CFG.chunk_size, ACTION_DIM)
    assert mu.shape == (4, TINY_MODEL_CFG.z_dim)
    assert logvar.shape == (4, TINY_MODEL_CFG.z_dim)
    assert torch.isfinite(a_hat).all()


def test_predict_is_deterministic_with_zero_latent() -> None:
    model = _tiny_model()
    model.eval()
    obs = _tiny_batch()["obs"]
    first = model.predict(obs)
    second = model.predict(obs)

    assert first.shape == (4, TINY_MODEL_CFG.chunk_size, ACTION_DIM)
    assert torch.equal(first, second)


def test_forward_rejects_wrong_chunk_length() -> None:
    model = _tiny_model()
    batch = _tiny_batch()
    with pytest.raises(ValueError, match="expected actions of shape"):
        model(batch["obs"], batch["actions"][:, :-1], batch["is_pad"][:, :-1])


def test_masked_l1_ignores_padded_slots() -> None:
    batch = _tiny_batch()
    a_hat = torch.zeros_like(batch["actions"])
    mu = torch.zeros(4, TINY_MODEL_CFG.z_dim)
    logvar = torch.zeros(4, TINY_MODEL_CFG.z_dim)

    base = act_loss(a_hat, batch["actions"], batch["is_pad"], mu, logvar, kl_weight=1.0)
    corrupted = batch["actions"].clone()
    corrupted[batch["is_pad"]] = 1e6
    altered = act_loss(a_hat, corrupted, batch["is_pad"], mu, logvar, kl_weight=1.0)

    assert base["l1"].item() == pytest.approx(altered["l1"].item())
    assert base["loss"].item() == pytest.approx(altered["loss"].item())


def test_kl_term_is_zero_at_prior_and_positive_away_from_it() -> None:
    batch = _tiny_batch()
    a_hat = batch["actions"].clone()
    zeros = torch.zeros(4, TINY_MODEL_CFG.z_dim)

    at_prior = act_loss(a_hat, batch["actions"], batch["is_pad"], zeros, zeros, kl_weight=1.0)
    assert at_prior["kl"].item() == pytest.approx(0.0)
    assert at_prior["l1"].item() == pytest.approx(0.0)

    off_prior = act_loss(
        a_hat, batch["actions"], batch["is_pad"], zeros + 2.0, zeros, kl_weight=1.0
    )
    assert off_prior["kl"].item() > 0.0
    assert off_prior["loss"].item() == pytest.approx(off_prior["kl"].item())


def test_all_padded_batch_is_rejected() -> None:
    batch = _tiny_batch()
    zeros = torch.zeros(4, TINY_MODEL_CFG.z_dim)
    with pytest.raises(ValueError, match="all-padded"):
        act_loss(
            batch["actions"],
            batch["actions"],
            torch.ones_like(batch["is_pad"]),
            zeros,
            zeros,
            kl_weight=1.0,
        )


def test_checkpoint_round_trip_preserves_predictions_and_stats(tmp_path) -> None:
    model = _tiny_model()
    model.eval()
    obs = _tiny_batch()["obs"]
    expected = model.predict(obs)
    stats = _tiny_stats()
    path = _save_checkpoint(tmp_path / "best.pt", model, _checkpoint_config(), stats)
    policy = ActPolicy.from_checkpoint(path, runtime_asset=TEST_ROBOT_ASSET)

    assert torch.equal(policy.model.predict(obs), expected)
    assert policy.action_space == "A2_ee_delta"
    assert policy.obs_preset == "core"
    assert policy.chunk_size == TINY_MODEL_CFG.chunk_size
    assert policy.robot_compatibility_label == "v2_native"

    for name in ("mean", "std", "min", "max"):
        np.testing.assert_array_equal(
            getattr(policy.stats.action, name), getattr(stats.action, name)
        )
        np.testing.assert_array_equal(getattr(policy.stats.obs, name), getattr(stats.obs, name))

    with pytest.raises(ValueError, match="incompatible"):
        ActPolicy.from_checkpoint(
            path,
            runtime_asset=RobotAssetRef("different_alex_v2", "b" * 64),
        )


def test_checkpoint_creation_rejects_config_stats_preset_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="observation preset"):
        _save_checkpoint(
            tmp_path / "bad.pt",
            _tiny_model(),
            _checkpoint_config("core_door_pose"),
            _tiny_stats(),
        )


def test_checkpoint_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format": "other"}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        ActPolicy.from_checkpoint(path, runtime_asset=TEST_ROBOT_ASSET)


# --- training loop -----------------------------------------------------------


def _constant_mapping_batch(batch: int = 8) -> dict[str, np.ndarray]:
    """A deterministic obs -> chunk mapping the tiny model must overfit."""
    horizon = TINY_MODEL_CFG.chunk_size
    obs = np.tile(np.linspace(-1.0, 1.0, OBS_DIM), (batch, 1))
    ramp = np.linspace(-0.5, 0.5, horizon).reshape(1, horizon, 1)
    actions = np.tile(ramp, (batch, 1, ACTION_DIM))
    return {
        "obs": obs,
        "actions": actions,
        "is_pad": np.zeros((batch, horizon), dtype=bool),
    }


def test_train_act_overfits_a_constant_mapping() -> None:
    model = _tiny_model()
    batch = _constant_mapping_batch()
    cfg = ActTrainCfg(
        epochs=200,
        batch_size=8,
        lr=1e-3,
        kl_weight=1.0,
        seed=0,
        val_every=50,
        device="cpu",
    )

    history = train_act(
        model,
        make_train_batches=lambda epoch: [batch],
        cfg=cfg,
        make_val_batches=lambda: [batch],
    )

    assert len(history.epochs) == cfg.epochs
    first, last = history.epochs[0], history.epochs[-1]
    assert last.train_l1 < 0.3 * first.train_l1
    assert last.train_l1 < 0.15
    assert last.val_l1 is not None and math.isfinite(last.val_l1)
    assert 0 <= history.best_epoch < cfg.epochs
    assert history.best_val_l1 <= last.val_l1 + 1e-12


def test_train_act_resume_matches_uninterrupted_state() -> None:
    batch = _constant_mapping_batch(batch=4)
    cfg = ActTrainCfg(
        epochs=4,
        batch_size=4,
        lr=1e-3,
        seed=17,
        val_every=1,
        device="cpu",
    )
    full_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    full_history = train_act(full_model, lambda epoch: [batch], cfg)

    interrupted_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    captured: dict = {}

    class StopAfterEpoch(RuntimeError):
        pass

    def interrupt_after_two_epochs(state) -> None:
        if state["next_epoch"] == 2:
            captured["training_state"] = deepcopy(state)
            captured["model_state"] = {
                key: value.detach().clone() for key, value in interrupted_model.state_dict().items()
            }
            raise StopAfterEpoch

    with pytest.raises(StopAfterEpoch):
        train_act(
            interrupted_model,
            lambda epoch: [batch],
            cfg,
            on_checkpoint=interrupt_after_two_epochs,
        )

    resumed_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    resumed_model.load_state_dict(captured["model_state"])
    resumed_history = train_act(
        resumed_model,
        lambda epoch: [batch],
        cfg,
        resume_state=captured["training_state"],
    )
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert [entry.train_loss for entry in resumed_history.epochs] == pytest.approx(
        [entry.train_loss for entry in full_history.epochs]
    )


# --- policy wrapper ----------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Captures the normalized obs it receives; predicts a fixed chunk."""

    def __init__(self, output_value: float = 1.0) -> None:
        super().__init__()
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.cfg = TINY_MODEL_CFG
        self.output_value = output_value
        self.last_input: torch.Tensor | None = None

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        self.last_input = obs.detach().clone()
        return torch.full((obs.shape[0], self.cfg.chunk_size, self.action_dim), self.output_value)


def _identity_obs_stats() -> NormStats:
    return NormStats(
        mean=np.zeros(OBS_DIM),
        std=np.ones(OBS_DIM),
        min=np.zeros(OBS_DIM),
        max=np.zeros(OBS_DIM),
        count=1,
    )


def test_act_policy_normalizes_input_and_denormalizes_output() -> None:
    action_mean = np.arange(ACTION_DIM, dtype=np.float64)
    action_std = np.full(ACTION_DIM, 0.5)
    obs_mean = np.linspace(1.0, 2.0, OBS_DIM)
    obs_std = np.full(OBS_DIM, 2.0)
    stats = DatasetNormStats(
        action=NormStats(action_mean, action_std, action_mean, action_mean, 1),
        obs=NormStats(obs_mean, obs_std, obs_mean, obs_mean, 1),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )
    model = _StubModel(output_value=1.0)
    policy = ActPolicy(model, stats)

    chunk = policy.predict(obs_mean)  # obs at the mean -> normalized zeros

    assert model.last_input is not None
    np.testing.assert_allclose(model.last_input.numpy(), np.zeros((1, OBS_DIM)), atol=1e-6)
    assert chunk.shape == (TINY_MODEL_CFG.chunk_size, ACTION_DIM)
    expected = np.tile(action_std * 1.0 + action_mean, (TINY_MODEL_CFG.chunk_size, 1))
    np.testing.assert_allclose(chunk, expected, atol=1e-6)


def test_act_policy_clips_exploding_normalized_obs() -> None:
    stats = DatasetNormStats(
        action=NormStats(
            np.zeros(ACTION_DIM), np.ones(ACTION_DIM), np.zeros(ACTION_DIM), np.zeros(ACTION_DIM), 1
        ),
        obs=NormStats(
            np.zeros(OBS_DIM), np.full(OBS_DIM, 1e-8), np.zeros(OBS_DIM), np.zeros(OBS_DIM), 1
        ),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )
    model = _StubModel()
    policy = ActPolicy(model, stats)

    policy.predict(np.full(OBS_DIM, 1e-3))  # would normalize to 1e5 without the clip

    assert model.last_input is not None
    assert float(model.last_input.abs().max()) == pytest.approx(policy.obs_clip)


def test_act_policy_rejects_mismatched_stats() -> None:
    stats = _tiny_stats()
    model = ACTModel(obs_dim=OBS_DIM + 1, action_dim=ACTION_DIM, cfg=TINY_MODEL_CFG)
    with pytest.raises(ValueError, match="obs dim"):
        ActPolicy(model, stats)


# --- rollout observation builder ---------------------------------------------


def _step_context(env):
    return read_step_context(env, read_door_frame(env))


def test_build_rollout_obs_matches_validated_context() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    ctx = _step_context(env)
    obs = build_rollout_obs(ctx, "core")

    expected = np.concatenate(
        [
            ctx.ee_pos_w,
            ctx.ee_quat_w_xyzw,
            [ctx.hinge_angle_rad, ctx.hinge_velocity_rad_s],
        ]
    )
    assert obs.shape == (9,)
    np.testing.assert_allclose(obs, expected)


def test_build_rollout_obs_core_contact_uses_sensor_state() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    obs = build_rollout_obs(_step_context(env), "core_contact")
    assert obs.shape == (10,)
    assert obs[-1] in (0.0, 1.0)


def test_build_rollout_obs_rejects_unknown_presets() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    with pytest.raises(ValueError, match="unknown obs preset"):
        build_rollout_obs(_step_context(env), "unsupported")


def test_build_rollout_obs_core_door_pose_matches_dataset_ordering() -> None:
    from alexdoor_xas.data_engine import plan_episodes, run_episode
    from alexdoor_xas.dataset.loader import EpisodeRecord, obs_matrix

    yaw = 0.45
    origin = (0.8, -1.5, 0.4)

    live_env = FakeDoorPushEnv(yaw_rad=yaw, origin=origin)
    live_env.reset(seed=0)
    live = build_rollout_obs(
        _step_context(live_env),
        "core_door_pose",
        read_door_pose_obs(live_env),
    )
    assert live.shape == (14,)
    np.testing.assert_allclose(live[9:12], origin, atol=1e-12)
    np.testing.assert_allclose(live[12], np.sin(yaw), atol=1e-12)
    np.testing.assert_allclose(live[13], np.cos(yaw), atol=1e-12)

    # Same env recorded through the data engine: step-0 obs must match the
    # live reader on a freshly reset env (both capture the pre-step state).
    episode = run_episode(
        FakeDoorPushEnv(yaw_rad=yaw, origin=origin),
        plan_episodes(1, 0, 0)[0],
        make_test_engine_cfg(),
    )
    obs = {}
    for table in ("proprio", "object_state", "contact"):
        for key, first in getattr(episode.steps[0], table).items():
            if isinstance(first, str):
                continue
            obs[key] = np.asarray(
                [getattr(step, table)[key] for step in episode.steps], dtype=np.float64
            )
    record = EpisodeRecord(
        episode_id="parity",
        action_space=episode.meta.action_space,
        schema_version="phase2.v2",
        meta=episode.meta.to_dict(),
        t=np.array([step.t for step in episode.steps]),
        actions=episode.stacked(lambda s: s.action),
        obs=obs,
        success=True,
        final_door_angle=1.0,
        termination_reason="controller_done",
        environment_terminated=False,
        environment_truncated=False,
        extras=episode.extras,
        buffer=episode,
    )
    np.testing.assert_allclose(obs_matrix(record, "core_door_pose")[0], live, atol=1e-9)


# --- chunk source + adapter rollout (closed-loop smoke, no Isaac) -------------


def _rollout_policy() -> ActPolicy:
    """Tiny real model whose denormalized deltas stay within the A2 clamps."""
    stats = DatasetNormStats(
        action=NormStats(
            np.zeros(ACTION_DIM),
            np.full(ACTION_DIM, 1e-3),
            np.zeros(ACTION_DIM),
            np.zeros(ACTION_DIM),
            1,
        ),
        obs=_identity_obs_stats(),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )
    return ActPolicy(_tiny_model(), stats)


def test_act_chunk_source_drives_a2_adapter_rollout() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    policy = _rollout_policy()
    source = act_chunk_source(policy, env)
    adapter = A2Adapter(TEST_ROBOT_LIMITS)

    result = rollout_chunks(env, source, adapter, max_ticks=20)

    assert result.n_ticks == 20
    assert len(result.decisions_per_tick) == 20
    assert len(adapter.log.decisions) == 20
    assert math.isfinite(result.final_angle_rad)

    chunk = policy.predict(np.zeros(OBS_DIM))
    assert np.isfinite(chunk).all()
    assert np.abs(chunk[:, :3]).max() < 0.04


class _QueuePolicy:
    """Duck-typed policy stub emitting predetermined chunks."""

    obs_preset = "core"

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = list(chunks)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return self._chunks.pop(0)


def test_temporal_ensemble_weights_match_the_paper_scheme() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    m = 0.5
    chunk_a = np.tile(np.array([[1.0, 0, 0, 0, 0, 0]]), (3, 1)) * np.array([[1], [2], [3]])
    chunk_b = np.tile(np.array([[10.0, 0, 0, 0, 0, 0]]), (3, 1))
    source = act_chunk_source(
        _QueuePolicy([chunk_a, chunk_b]), env, temporal_ensemble=True, ensemble_m=m
    )
    ctx = _step_context(env)

    first = source(ctx)
    assert first.shape == (1, 6)
    np.testing.assert_allclose(first[0], chunk_a[0])

    second = source(ctx)
    weights = np.array([1.0, math.exp(-m)])  # oldest chunk first, weight exp(-m * i)
    expected = (chunk_a[1] * weights[0] + chunk_b[0] * weights[1]) / weights.sum()
    np.testing.assert_allclose(second[0], expected)


# --- open-loop inspection ------------------------------------------------------


def _stub_record(n_steps: int = 12, episode_id: str = "ep-stub-0001") -> EpisodeRecord:
    actions = np.zeros((n_steps, ACTION_DIM))
    actions[:, 0] = np.linspace(0.0, 0.011, n_steps)
    return EpisodeRecord(
        episode_id=episode_id,
        action_space="A2_ee_delta",
        schema_version="phase2.v1",
        meta={},
        t=np.arange(n_steps, dtype=np.float64) / 60.0,
        actions=actions,
        obs={
            "ee_pos_w": np.zeros((n_steps, 3)),
            "ee_quat_w_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (n_steps, 1)),
            "door_angle_rad": np.zeros(n_steps),
            "door_angular_velocity_rad_s": np.zeros(n_steps),
        },
        success=True,
        final_door_angle=0.8,
        termination_reason="controller_done",
        environment_terminated=False,
        environment_truncated=False,
        extras={},
        buffer=None,
    )


class _OffsetPolicy:
    """Predicts the recorded action plus a fixed offset on every dim."""

    obs_preset = "core"
    action_space = "A2_ee_delta"
    chunk_size = TINY_MODEL_CFG.chunk_size

    def __init__(self, record: EpisodeRecord, offset: float) -> None:
        self._record = record
        self._offset = offset
        self.stats = _tiny_stats()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        del obs
        chunk = np.full((self.chunk_size, ACTION_DIM), self._offset)
        chunk += self._record.actions[: self.chunk_size]
        return chunk


def test_open_loop_report_numerics(tmp_path) -> None:
    record = _stub_record(n_steps=TINY_MODEL_CFG.chunk_size)  # one exact chunk
    policy = _OffsetPolicy(record, offset=0.25)
    json_path = tmp_path / "metrics" / "open_loop.json"

    report = open_loop_report(policy, [record], json_path=json_path)

    assert json_path.is_file()
    assert report["evaluated_steps"] == record.n_steps
    assert report["aggregate_l1_mean"] == pytest.approx(0.25)
    assert report["l1_by_dimension"] == pytest.approx({"dx": 0.25, "dy": 0.25, "dz": 0.25})
    assert report["per_episode"] == [
        {
            "episode_id": record.episode_id,
            "l1_mean": pytest.approx(0.25),
            "evaluated_steps": record.n_steps,
        }
    ]
    assert "mse" not in json_path.read_text().lower()
