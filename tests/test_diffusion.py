"""Pure tests for the Diffusion Policy baseline (Phase 3.3). No Isaac imports.

Torch is required (as in ``test_act.py``); ``diffusers`` is skipped-if-missing
so a bare environment degrades gracefully instead of erroring at collection.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from alexdoor_xas.dataset import DatasetNormStats, NormStats  # noqa: E402
from alexdoor_xas.policies.diffusion.config import (  # noqa: E402
    DiffusionConfigError,
    DiffusionModelCfg,
)
from alexdoor_xas.policies.diffusion.data import (  # noqa: E402
    MinMaxNormalizer,
    make_diffusion_normalizer,
)
from alexdoor_xas.policies.diffusion.schedulers import (  # noqa: E402
    make_inference_scheduler,
    make_train_scheduler,
    sample_actions,
    scheduler_config_payload,
)

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
