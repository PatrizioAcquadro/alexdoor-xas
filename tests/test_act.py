"""Pure tests for the ACT baseline package (Phase 3.2). No Isaac imports."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alexdoor_xas.dataset import DatasetNormStats, NormStats
from alexdoor_xas.policies.act.checkpoint import load_checkpoint, save_checkpoint
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel, act_loss, sinusoidal_table

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
        dataset_episode_ids=("ep0", "ep1"),
        action_space="A2_ee_delta",
        dataset_fingerprint="deadbeef",
    )


def test_sinusoidal_table_shape_and_range() -> None:
    table = sinusoidal_table(10, 32)
    assert table.shape == (10, 32)
    assert torch.isfinite(table).all()
    assert table.abs().max() <= 1.0


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
    config = {"dataset": {"space": "A2_ee_delta", "obs_preset": "core"}}

    path = save_checkpoint(
        tmp_path / "ckpt" / "best.pt", model, config, stats, meta={"epoch": 3}
    )
    loaded = load_checkpoint(path)

    assert torch.equal(loaded.model.predict(obs), expected)
    assert loaded.action_space == "A2_ee_delta"
    assert loaded.obs_preset == "core"
    assert loaded.chunk_size == TINY_MODEL_CFG.chunk_size
    assert loaded.meta["epoch"] == 3
    assert not loaded.model.training

    for name in ("mean", "std", "min", "max"):
        np.testing.assert_array_equal(
            getattr(loaded.stats.action, name), getattr(stats.action, name)
        )
        np.testing.assert_array_equal(getattr(loaded.stats.obs, name), getattr(stats.obs, name))
    assert loaded.stats.train_episode_ids == stats.train_episode_ids
    assert loaded.stats.dataset_fingerprint == stats.dataset_fingerprint


def test_checkpoint_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format": "other"}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path)
