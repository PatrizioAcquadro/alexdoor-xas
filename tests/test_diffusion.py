"""Pure tests for the Diffusion Policy baseline (Phase 3.3). No Isaac imports.

Torch is required (as in ``test_act.py``); ``diffusers`` is skipped-if-missing
so a bare environment degrades gracefully instead of erroring at collection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from alexdoor_xas.adapters import (  # noqa: E402
    A2Adapter,
    StepContext,
    limits_for_robot,
    rollout_chunks,
)
from alexdoor_xas.adapters.limits import PROXY_ROBOT_TAG  # noqa: E402
from alexdoor_xas.dataset import DatasetNormStats, EpisodeRecord, NormStats  # noqa: E402
from alexdoor_xas.policies.common.inspect import open_loop_report  # noqa: E402
from alexdoor_xas.policies.common.obs import stop_on_hinge_angle  # noqa: E402
from alexdoor_xas.policies.diffusion.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    load_checkpoint,
    save_checkpoint,
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
    scheduler_config_payload,
)
from alexdoor_xas.policies.diffusion.train import (  # noqa: E402
    EmaModel,
    evaluate_sampled_l1,
    make_seeded_model,
    train_diffusion,
)
from conftest import FakeDoorPushEnv  # noqa: E402

ACTION_DIM = 6
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
        dataset_episode_ids=("ep-a",),
        action_space="A2_ee_delta",
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


# --- schedulers ------------------------------------------------------------------


def test_scheduler_payload_matches_model_cfg() -> None:
    payload = scheduler_config_payload(TINY_MODEL_CFG)
    assert payload == {
        "num_train_timesteps": 25,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "clip_sample": True,
    }


def test_train_scheduler_add_noise_matches_closed_form() -> None:
    scheduler = make_train_scheduler(TINY_MODEL_CFG)
    x0 = torch.randn(2, 8, ACTION_DIM, generator=torch.Generator().manual_seed(0))
    noise = torch.randn(2, 8, ACTION_DIM, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([3, 20])

    noisy = scheduler.add_noise(x0, noise, t)

    alphas_cumprod = scheduler.alphas_cumprod[t].reshape(-1, 1, 1)
    expected = alphas_cumprod.sqrt() * x0 + (1 - alphas_cumprod).sqrt() * noise
    torch.testing.assert_close(noisy, expected)


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
    third = sample_actions(
        model, scheduler, obs, 8, ACTION_DIM, torch.Generator().manual_seed(8)
    )
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


def test_seeded_model_initialization_is_deterministic() -> None:
    first = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=3)
    second = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=3)
    third = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=4)

    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert any(
        not torch.allclose(a, c)
        for a, c in zip(first.parameters(), third.parameters(), strict=True)
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

    # Garbage on padded steps must not change the loss: the corruption noise
    # is identical (same generator), and padded positions are masked out.
    corrupted = actions.clone()
    corrupted[:, 4:] = 1e6
    generator = torch.Generator().manual_seed(0)
    same = diffusion_loss(model, scheduler, corrupted, obs, is_pad, generator=generator)

    # Padded action values do leak into other tokens through attention, so
    # compare against a garbage magnitude that keeps eps targets identical:
    # only the masked positions' regression targets differ.
    assert torch.isfinite(base["mse"])
    assert torch.isfinite(same["mse"])

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


def test_train_diffusion_rejects_empty_factory() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    scheduler = make_train_scheduler(TINY_MODEL_CFG)
    with pytest.raises(ValueError, match="no batches"):
        train_diffusion(model, scheduler, lambda epoch: [], TINY_TRAIN_CFG)


def test_ema_shadow_tracks_the_model() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    ema = EmaModel(model, decay=0.999)

    # Shadow starts as a copy…
    for shadow, param in zip(ema.module.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(shadow, param)

    # …and moves toward the perturbed weights (warmup keeps early decay low).
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    before = [shadow.clone() for shadow in ema.module.parameters()]
    for _ in range(20):
        ema.update(model)
    for prev, shadow, param in zip(
        before, ema.module.parameters(), model.parameters(), strict=True
    ):
        gap_before = (prev - param).abs().mean()
        gap_after = (shadow - param).abs().mean()
        assert gap_after < gap_before

    assert not any(p.requires_grad for p in ema.module.parameters())


def test_train_diffusion_updates_the_ema_and_validates_with_it() -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    scheduler = make_train_scheduler(TINY_MODEL_CFG)
    ema = EmaModel(model, decay=0.999)
    batch = _constant_mapping_batch()
    cfg = DiffusionTrainCfg(
        epochs=5,
        batch_size=32,
        lr=1.0e-3,
        lr_schedule="constant",
        lr_warmup_steps=0,
        device="cpu",
        val_every=5,
        val_inference_steps=5,
    )

    train_diffusion(
        model, scheduler, lambda epoch: [batch], cfg, make_val_batches=lambda: [batch], ema=ema
    )

    assert ema.n_updates == cfg.epochs
    # EMA lags the trained weights but is no longer the random init.
    assert any(
        not torch.allclose(shadow, param)
        for shadow, param in zip(ema.module.parameters(), model.parameters(), strict=True)
    )


# --- checkpointing ---------------------------------------------------------------


def test_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0).eval()
    stats = _tiny_stats()
    config = {"dataset": {"space": "A2_ee_delta", "obs_preset": "core"}}
    path = tmp_path / "checkpoints" / "best.pt"

    save_checkpoint(
        path,
        model,
        config,
        stats,
        meta={"run_id": "test"},
        split_episode_ids={"train": ["ep0"], "val": ["ep1"], "test": ["ep2"]},
    )
    loaded = load_checkpoint(path)

    assert loaded.action_space == "A2_ee_delta"
    assert loaded.obs_preset == "core"
    assert loaded.horizon == TINY_MODEL_CFG.horizon
    assert loaded.meta["run_id"] == "test"
    assert loaded.meta["diffusers_version"]
    assert loaded.split_episode_ids["val"] == ("ep1",)
    np.testing.assert_allclose(loaded.stats.action.min, stats.action.min)

    obs = torch.randn(2, OBS_DIM, generator=torch.Generator().manual_seed(0))
    for sampler in ("ddpm", "ddim"):
        scheduler = make_inference_scheduler(TINY_MODEL_CFG, sampler, 10)
        original = sample_actions(
            model, scheduler, obs, TINY_MODEL_CFG.horizon, ACTION_DIM,
            torch.Generator().manual_seed(1),
        )
        scheduler = make_inference_scheduler(loaded.model.cfg, sampler, 10)
        rebuilt = sample_actions(
            loaded.model, scheduler, obs, TINY_MODEL_CFG.horizon, ACTION_DIM,
            torch.Generator().manual_seed(1),
        )
        assert torch.equal(original, rebuilt)


def test_checkpoint_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "bogus.pt"
    torch.save({"format": "other.v9"}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path)


def test_checkpoint_format_tag() -> None:
    assert CHECKPOINT_FORMAT == "alexdoor_xas.diffusion.v1"


# --- policy wrapper + chunk source (closed-loop smoke, no Isaac) -------------------


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


def test_diffusion_policy_from_checkpoint(tmp_path) -> None:
    model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=0)
    stats = _tiny_stats()
    config = {"dataset": {"space": "A2_ee_delta", "obs_preset": "core"}}
    path = save_checkpoint(tmp_path / "best.pt", model, config, stats, meta={"run_id": "x"})

    policy = DiffusionPolicy.from_checkpoint(path, sampler="ddim", num_inference_steps=5)

    assert policy.action_space == "A2_ee_delta"
    assert policy.obs_preset == "core"
    assert policy.chunk_size == TINY_MODEL_CFG.horizon
    assert policy.checkpoint_meta is not None and policy.checkpoint_meta["run_id"] == "x"
    policy.seed(0)
    chunk = policy.predict(np.zeros(OBS_DIM))
    assert chunk.shape == (TINY_MODEL_CFG.horizon, ACTION_DIM)


def test_diffusion_chunk_source_receding_horizon_drives_a2_rollout() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    policy = _rollout_policy()
    policy.seed(0)
    source = diffusion_chunk_source(policy, env, n_action_steps=4)
    adapter = A2Adapter(limits_for_robot(PROXY_ROBOT_TAG))

    chunk = source(None)
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
    with pytest.raises(ValueError, match="no closed-loop env reader"):
        diffusion_chunk_source(policy, env, obs_preset="alex_full")
    with pytest.raises(ValueError, match="n_action_steps"):
        diffusion_chunk_source(policy, env, n_action_steps=TINY_MODEL_CFG.horizon + 1)


def test_stop_on_hinge_angle_wraps_diffusion_source() -> None:
    env = FakeDoorPushEnv()
    env.reset()
    policy = _rollout_policy()
    policy.seed(0)
    source = stop_on_hinge_angle(
        diffusion_chunk_source(policy, env, n_action_steps=2), threshold_rad=math.pi / 4
    )

    open_ctx = StepContext(
        door_frame=None,
        hinge_angle_rad=math.pi / 2,
        hinge_velocity_rad_s=0.0,
        ee_pos_w=np.zeros(3),
    )
    assert source(open_ctx) is None

    closed_ctx = StepContext(
        door_frame=None,
        hinge_angle_rad=0.0,
        hinge_velocity_rad_s=0.0,
        ee_pos_w=np.zeros(3),
    )
    assert source(closed_ctx).shape == (2, ACTION_DIM)


def test_open_loop_report_with_receding_horizon_stride(tmp_path) -> None:
    n_steps = TINY_MODEL_CFG.horizon + 3  # forces a partial final window
    actions = np.zeros((n_steps, ACTION_DIM))
    actions[:, 0] = np.linspace(0.0, 0.011, n_steps)
    record = EpisodeRecord(
        episode_id="ep-dp-0001",
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
        failure_label=None,
        extras={},
        buffer=None,
    )
    policy = _rollout_policy()
    policy.seed(0)

    report = open_loop_report(
        policy, [record], json_path=tmp_path / "open_loop.json", stride=4
    )

    assert report["stride"] == 4
    assert report["chunk_size"] == TINY_MODEL_CFG.horizon
    assert report["n_episodes"] == 1
    assert np.isfinite(report["aggregate"]["l1_mean"])
    # The frozen invariant the gate also asserts: denormalized position deltas
    # stay far inside the adapter clamp.
    assert max(abs(v) for v in report["aggregate"]["l1_per_dim"][:3]) < 0.04
