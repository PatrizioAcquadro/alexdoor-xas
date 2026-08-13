"""Diffusion policy contracts without Isaac imports."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from alexdoor_xas.adapters.a2 import A2Adapter  # noqa: E402
from alexdoor_xas.adapters.rollout import (  # noqa: E402
    read_door_frame,
    read_step_context,
    rollout_chunks,
)
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef  # noqa: E402
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats  # noqa: E402
from alexdoor_xas.policies.common.checkpoint import (  # noqa: E402
    DIFFUSION_CHECKPOINT_FORMAT,
    save_checkpoint_payload,
)
from alexdoor_xas.policies.diffusion.config import (  # noqa: E402
    DiffusionConfigError,
    DiffusionModelCfg,
    DiffusionTrainCfg,  # noqa: E402
)
from alexdoor_xas.policies.diffusion.data import (  # noqa: E402
    MinMaxNormalizer,
    make_diffusion_normalizer,
)
from alexdoor_xas.policies.diffusion.model import (  # noqa: E402
    diffusion_loss,
)
from alexdoor_xas.policies.diffusion.policy import (  # noqa: E402
    DiffusionPolicy,
    diffusion_chunk_source,
)
from alexdoor_xas.policies.diffusion.schedulers import (  # noqa: E402
    make_inference_scheduler,
    make_train_scheduler,
    sample_actions,
)
from alexdoor_xas.policies.diffusion.train import (  # noqa: E402
    EmaModel,
    evaluate_sampled_l1,
    make_seeded_model,
    train_diffusion,
)
from conftest import TEST_ROBOT_LIMITS, FakeDoorPushEnv  # noqa: E402

ACTION_DIM = 6
TEST_ROBOT_ASSET = RobotAssetRef("alex_v2_test", "a" * 64)
OBS_DIM = 9

TINY_MODEL_CFG = DiffusionModelCfg(
    horizon=8,
    d_model=32,
    n_heads=2,
    n_decoder_layers=1,
    dim_feedforward=64,
    dropout=0.0,
    num_train_timesteps=25,
)


def _tiny_action_stats() -> NormStats:
    # Position dims span a real range; rotation dims are constant zero,
    # matching the frozen A2/A3 export.
    low = np.array([-0.015, -0.01, 0.0, 0.0, 0.0, 0.0])
    high = np.array([0.005, 0.013, 0.015, 0.0, 0.0, 0.0])
    return NormStats(
        mean=(low + high) / 2.0,
        std=np.maximum((high - low) / 4.0, 1e-8),
        min=low,
        max=high,
        count=100,
    )


def _tiny_stats() -> DatasetNormStats:
    obs = NormStats(
        mean=np.zeros(OBS_DIM),
        std=np.full(OBS_DIM, 0.5),
        min=np.full(OBS_DIM, -1.0),
        max=np.full(OBS_DIM, 1.0),
        count=100,
    )
    return DatasetNormStats(
        action=_tiny_action_stats(),
        obs=obs,
        obs_preset="core",
        train_episode_ids=("ep-a",),
        action_space="A2_ee_delta",
    )


def _checkpoint_config() -> dict:
    return {
        "dataset": {
            "task": "door_push_alex_v2",
            "space": "A2_ee_delta",
            "version": "v2_pose",
            "view_id": None,
            "obs_preset": "core",
        }
    }


def _save_checkpoint(path, model, stats):
    return save_checkpoint_payload(
        path,
        DIFFUSION_CHECKPOINT_FORMAT,
        model,
        _checkpoint_config(),
        stats,
        {},
        TEST_ROBOT_ASSET,
    )


# --- min-max normalization -------------------------------------------------------


def test_minmax_round_trip_and_extrema() -> None:
    stats = _tiny_action_stats()
    normalizer = MinMaxNormalizer.from_norm_stats(stats)

    rng = np.random.default_rng(0)
    x = rng.uniform(stats.min, stats.max, size=(50, ACTION_DIM))
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(x)), x, atol=1e-12)

    # Train extrema map to exactly ±1 on non-constant dims.
    np.testing.assert_allclose(normalizer.normalize(stats.min)[:3], -1.0)
    np.testing.assert_allclose(normalizer.normalize(stats.max)[:3], 1.0)
    # Everything inside the train range stays within [-1, 1].
    assert np.abs(normalizer.normalize(x)).max() <= 1.0 + 1e-12


def test_minmax_constant_dims_shift_to_exact_zero() -> None:
    normalizer = MinMaxNormalizer.from_norm_stats(_tiny_action_stats())

    zeros = np.zeros(ACTION_DIM)
    assert (normalizer.normalize(zeros)[3:] == 0.0).all()
    # A sampled (clipped) normalized value denormalizes through scale 1.0 —
    # no division by a floored std, no 1e8 blowup.
    sampled = np.array([0.5, -0.5, 0.25, 0.9, -0.9, 0.1])
    denorm = normalizer.denormalize(sampled)
    assert np.abs(denorm[3:]).max() <= 1.0  # bounded, not exploded
    assert np.isfinite(denorm).all()


def test_diffusion_batch_normalizer_mixes_zscore_obs_and_minmax_actions() -> None:
    stats = _tiny_stats()
    normalize = make_diffusion_normalizer(stats)
    batch = {
        "obs": np.full((4, OBS_DIM), 0.5),
        "actions": np.tile(stats.action.max, (4, TINY_MODEL_CFG.horizon, 1)),
        "is_pad": np.zeros((4, TINY_MODEL_CFG.horizon), dtype=bool),
    }

    out = normalize(batch, stats)

    np.testing.assert_allclose(out["obs"], 1.0)  # (0.5 - 0) / 0.5
    np.testing.assert_allclose(out["actions"][..., :3], 1.0)
    np.testing.assert_allclose(out["actions"][..., 3:], 0.0)
    assert out["is_pad"] is batch["is_pad"]  # passthrough


def test_inference_scheduler_validates_inputs() -> None:
    with pytest.raises(DiffusionConfigError, match="num_inference_steps"):
        make_inference_scheduler(TINY_MODEL_CFG, "ddpm", 26)
    with pytest.raises(DiffusionConfigError, match="sampler"):
        make_inference_scheduler(TINY_MODEL_CFG, "heun", 10)


class _ZeroEpsModel(torch.nn.Module):
    """Predicts zero noise: DDIM then contracts any start toward x0 = x_T."""

    def forward(self, x, t, obs):  # noqa: D102
        del t, obs
        return torch.zeros_like(x)


def test_sample_actions_is_deterministic_with_seeded_generator() -> None:
    model = _ZeroEpsModel()
    obs = torch.zeros(3, OBS_DIM)

    for sampler in ("ddpm", "ddim"):
        scheduler = make_inference_scheduler(TINY_MODEL_CFG, sampler, 10)
        first = sample_actions(
            model, scheduler, obs, 8, ACTION_DIM, torch.Generator().manual_seed(7)
        )
        scheduler = make_inference_scheduler(TINY_MODEL_CFG, sampler, 10)
        second = sample_actions(
            model, scheduler, obs, 8, ACTION_DIM, torch.Generator().manual_seed(7)
        )
        assert first.shape == (3, 8, ACTION_DIM)
        assert torch.isfinite(first).all()
        assert first.abs().max() <= 1.0 + 1e-6  # clip_sample bound
        torch.testing.assert_close(first, second)

    scheduler = make_inference_scheduler(TINY_MODEL_CFG, "ddpm", 10)
    third = sample_actions(model, scheduler, obs, 8, ACTION_DIM, torch.Generator().manual_seed(8))
    assert not torch.allclose(first, third)


# --- model + loss ----------------------------------------------------------------


def test_model_forward_shapes_and_finiteness() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    x = torch.randn(4, TINY_MODEL_CFG.horizon, ACTION_DIM)
    t = torch.tensor([0, 5, 12, 24])
    obs = torch.randn(4, OBS_DIM)

    eps_hat = model(x, t, obs)

    assert eps_hat.shape == (4, TINY_MODEL_CFG.horizon, ACTION_DIM)
    assert torch.isfinite(eps_hat).all()
    assert model.n_parameters > 0


def test_model_output_depends_on_obs_and_timestep() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0).eval()
    x = torch.randn(1, TINY_MODEL_CFG.horizon, ACTION_DIM)
    obs = torch.randn(1, OBS_DIM)

    base = model(x, torch.tensor([3]), obs)
    other_t = model(x, torch.tensor([20]), obs)
    other_obs = model(x, torch.tensor([3]), obs + 1.0)

    assert not torch.allclose(base, other_t)
    assert not torch.allclose(base, other_obs)


def test_model_causal_mask_blocks_future_tokens() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0).eval()
    x = torch.randn(1, TINY_MODEL_CFG.horizon, ACTION_DIM)
    obs = torch.randn(1, OBS_DIM)
    t = torch.tensor([3])

    base = model(x, t, obs)
    perturbed = x.clone()
    perturbed[0, -1] += 10.0  # only the last action token changes
    changed = model(perturbed, t, obs)

    # Every earlier output token attends only to itself and the past, so it
    # cannot see the perturbation of the final token.
    torch.testing.assert_close(base[0, :-1], changed[0, :-1])
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_model_rejects_bad_shapes() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    with pytest.raises(ValueError, match="noisy actions"):
        model(torch.randn(1, 3, ACTION_DIM), torch.tensor([0]), torch.randn(1, OBS_DIM))
    with pytest.raises(ValueError, match="obs"):
        model(
            torch.randn(1, TINY_MODEL_CFG.horizon, ACTION_DIM),
            torch.tensor([0]),
            torch.randn(1, OBS_DIM + 1),
        )


def test_diffusion_loss_masks_padded_steps() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    scheduler = make_train_scheduler(TINY_MODEL_CFG)
    actions = torch.randn(2, TINY_MODEL_CFG.horizon, ACTION_DIM)
    obs = torch.randn(2, OBS_DIM)
    is_pad = torch.zeros(2, TINY_MODEL_CFG.horizon, dtype=torch.bool)
    is_pad[:, 4:] = True

    generator = torch.Generator().manual_seed(0)
    base = diffusion_loss(model, scheduler, actions, obs, is_pad, generator=generator)

    corrupted = actions.clone()
    corrupted[:, 4:] = 1e6
    generator = torch.Generator().manual_seed(0)
    same = diffusion_loss(model, scheduler, corrupted, obs, is_pad, generator=generator)

    assert torch.isfinite(base["mse"])
    torch.testing.assert_close(base["mse"], same["mse"])

    with pytest.raises(ValueError, match="all-padded"):
        diffusion_loss(
            model,
            scheduler,
            actions,
            obs,
            torch.ones(2, TINY_MODEL_CFG.horizon, dtype=torch.bool),
        )


# --- training --------------------------------------------------------------------

TINY_TRAIN_CFG = DiffusionTrainCfg(
    epochs=150,
    batch_size=32,
    lr=1.0e-3,
    weight_decay=0.0,
    grad_clip=1.0,
    lr_schedule="constant",
    lr_warmup_steps=0,
    use_ema=False,
    seed=0,
    device="cpu",
    val_every=50,
    val_inference_steps=10,
)


def _constant_mapping_batch(batch_size: int = 32) -> dict:
    obs = np.tile(np.linspace(-1.0, 1.0, OBS_DIM), (batch_size, 1))
    actions = np.full((batch_size, TINY_MODEL_CFG.horizon, ACTION_DIM), 0.5)
    is_pad = np.zeros((batch_size, TINY_MODEL_CFG.horizon), dtype=bool)
    return {"obs": obs, "actions": actions, "is_pad": is_pad}


def test_train_diffusion_overfits_a_constant_mapping() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    scheduler = make_train_scheduler(TINY_MODEL_CFG)
    batch = _constant_mapping_batch()

    events: list[tuple[int, bool]] = []
    history = train_diffusion(
        model,
        scheduler,
        lambda epoch: [batch],
        TINY_TRAIN_CFG,
        make_val_batches=lambda: [batch],
        on_epoch=lambda stats, is_best: events.append((stats.epoch, is_best)),
    )

    assert len(history.epochs) == TINY_TRAIN_CFG.epochs
    assert history.epochs[-1].train_mse < 0.5 * history.epochs[0].train_mse
    assert history.best_epoch >= 0
    assert history.best_val_l1 < 0.25
    assert len(events) == TINY_TRAIN_CFG.epochs
    assert any(is_best for _, is_best in events)

    sampled_l1 = evaluate_sampled_l1(model, [batch], torch.device("cpu"), 10)
    assert sampled_l1 < 0.25


def test_train_diffusion_resume_matches_uninterrupted_state_and_ema() -> None:
    batch = _constant_mapping_batch(batch_size=4)
    cfg = DiffusionTrainCfg(
        epochs=4,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_warmup_steps=0,
        use_ema=True,
        ema_decay=0.99,
        seed=19,
        device="cpu",
        val_every=10,
        val_inference_steps=2,
    )
    full_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=19)
    full_ema = EmaModel(full_model, cfg.ema_decay)
    full_history = train_diffusion(
        full_model,
        make_train_scheduler(TINY_MODEL_CFG),
        lambda epoch: [batch],
        cfg,
        ema=full_ema,
    )

    interrupted_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=19)
    interrupted_ema = EmaModel(interrupted_model, cfg.ema_decay)
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
        train_diffusion(
            interrupted_model,
            make_train_scheduler(TINY_MODEL_CFG),
            lambda epoch: [batch],
            cfg,
            ema=interrupted_ema,
            on_checkpoint=interrupt_after_two_epochs,
        )

    resumed_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=19)
    resumed_model.load_state_dict(captured["model_state"])
    resumed_ema = EmaModel(resumed_model, cfg.ema_decay)
    resumed_history = train_diffusion(
        resumed_model,
        make_train_scheduler(TINY_MODEL_CFG),
        lambda epoch: [batch],
        cfg,
        ema=resumed_ema,
        resume_state=captured["training_state"],
    )
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
    for key, value in full_ema.module.state_dict().items():
        torch.testing.assert_close(value, resumed_ema.module.state_dict()[key], rtol=0, atol=0)
    assert resumed_ema.n_updates == full_ema.n_updates
    assert [entry.train_mse for entry in resumed_history.epochs] == pytest.approx(
        [entry.train_mse for entry in full_history.epochs]
    )


def test_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0).eval()
    stats = _tiny_stats()
    path = _save_checkpoint(tmp_path / "best.pt", model, stats)
    policy = DiffusionPolicy.from_checkpoint(
        path,
        sampler="ddim",
        num_inference_steps=5,
        runtime_asset=TEST_ROBOT_ASSET,
    )

    assert policy.action_space == "A2_ee_delta"
    assert policy.obs_preset == "core"
    assert policy.chunk_size == TINY_MODEL_CFG.horizon
    assert policy.robot_compatibility_label == "v2_native"
    np.testing.assert_allclose(policy.stats.action.min, stats.action.min)

    obs = torch.randn(2, OBS_DIM, generator=torch.Generator().manual_seed(0))
    for sampler in ("ddpm", "ddim"):
        scheduler = make_inference_scheduler(TINY_MODEL_CFG, sampler, 10)
        original = sample_actions(
            model,
            scheduler,
            obs,
            TINY_MODEL_CFG.horizon,
            ACTION_DIM,
            torch.Generator().manual_seed(1),
        )
        scheduler = make_inference_scheduler(policy.model.cfg, sampler, 10)
        rebuilt = sample_actions(
            policy.model,
            scheduler,
            obs,
            TINY_MODEL_CFG.horizon,
            ACTION_DIM,
            torch.Generator().manual_seed(1),
        )
        assert torch.equal(original, rebuilt)

    with pytest.raises(ValueError, match="incompatible"):
        DiffusionPolicy.from_checkpoint(
            path,
            sampler="ddim",
            num_inference_steps=5,
            runtime_asset=RobotAssetRef("different_alex_v2", "b" * 64),
        )


def _identity_obs_stats() -> NormStats:
    return NormStats(
        mean=np.zeros(OBS_DIM),
        std=np.ones(OBS_DIM),
        min=np.zeros(OBS_DIM),
        max=np.zeros(OBS_DIM),
        count=1,
    )


def _rollout_policy(**kwargs) -> DiffusionPolicy:
    """Tiny real model whose denormalized deltas stay within the A2 clamps."""
    stats = DatasetNormStats(
        action=_tiny_action_stats(),  # position range ±0.015 m < 0.02 m clamp
        obs=_identity_obs_stats(),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    return DiffusionPolicy(model, stats, num_inference_steps=5, **kwargs)


def test_diffusion_policy_predict_shape_and_bounds() -> None:
    policy = _rollout_policy()
    policy.seed(0)

    chunk = policy.predict(np.zeros(OBS_DIM))

    assert chunk.shape == (TINY_MODEL_CFG.horizon, ACTION_DIM)
    assert np.isfinite(chunk).all()
    # clip_sample + min-max denorm bound |dpos| by the train extrema — inside
    # the 0.02 m adapter clamp by construction.
    stats = _tiny_action_stats()
    assert (chunk[:, :3] >= stats.min[:3] - 1e-9).all()
    assert (chunk[:, :3] <= stats.max[:3] + 1e-9).all()
    # Constant rotation dims denormalize through scale 1.0 (bounded by ±1).
    assert np.abs(chunk[:, 3:]).max() <= 1.0


def test_diffusion_policy_obs_normalization_and_clip() -> None:
    stats = DatasetNormStats(
        action=_tiny_action_stats(),
        obs=NormStats(
            mean=np.zeros(OBS_DIM),
            std=np.full(OBS_DIM, 1e-8),
            min=np.zeros(OBS_DIM),
            max=np.zeros(OBS_DIM),
            count=1,
        ),
        obs_preset="core",
        train_episode_ids=("ep0",),
        action_space="A2_ee_delta",
    )

    captured: list[torch.Tensor] = []

    class _CaptureModel(torch.nn.Module):
        obs_dim = OBS_DIM
        action_dim = ACTION_DIM
        cfg = TINY_MODEL_CFG

        def forward(self, x, t, obs):  # noqa: D102
            captured.append(obs.detach().clone())
            return torch.zeros_like(x)

    policy = DiffusionPolicy(_CaptureModel(), stats, num_inference_steps=2)
    policy.seed(0)
    policy.predict(np.full(OBS_DIM, 1e-3))  # would normalize to 1e5 without the clip

    assert captured
    assert float(captured[0].abs().max()) == pytest.approx(policy.obs_clip)


def test_diffusion_policy_rejects_mismatched_stats() -> None:
    stats = _tiny_stats()
    model = make_seeded_model(OBS_DIM + 1, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    with pytest.raises(ValueError, match="obs dim"):
        DiffusionPolicy(model, stats)


def test_diffusion_policy_seed_makes_sampling_reproducible() -> None:
    policy = _rollout_policy()
    obs = np.zeros(OBS_DIM)

    policy.seed(11)
    first = policy.predict(obs)
    policy.seed(11)
    second = policy.predict(obs)
    policy.seed(12)
    third = policy.predict(obs)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_diffusion_chunk_source_receding_horizon_drives_a2_rollout() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    policy = _rollout_policy()
    policy.seed(0)
    source = diffusion_chunk_source(policy, env, n_action_steps=4)
    adapter = A2Adapter(TEST_ROBOT_LIMITS)

    chunk = source(read_step_context(env, read_door_frame(env)))
    assert chunk.shape == (4, ACTION_DIM)

    result = rollout_chunks(env, source, adapter, max_ticks=20)

    assert result.n_ticks == 20
    assert len(adapter.log.decisions) == 20
    assert math.isfinite(result.final_angle_rad)
    # Bounded-by-construction deltas: nothing to clamp, nothing to reject.
    assert adapter.log.count("corrected") == 0
    assert adapter.log.count("rejected") == 0


def test_diffusion_chunk_source_validates_inputs() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    policy = _rollout_policy()
    with pytest.raises(ValueError, match="unknown obs preset"):
        diffusion_chunk_source(policy, env, obs_preset="unsupported")
    with pytest.raises(ValueError, match="n_action_steps"):
        diffusion_chunk_source(policy, env, n_action_steps=TINY_MODEL_CFG.horizon + 1)
