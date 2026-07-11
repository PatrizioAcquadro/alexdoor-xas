#!/usr/bin/env python
"""Phase 3.3 training gate: Diffusion Policy learns the exported Phase 2 data (no Kit).

For each trainable action space (A2, A3) on the Alex task this verifies, with
a small model and a 2-episode overfit subset: the Hydra config composes; the
dataset/splits/norm-stats triple loads through the Phase 3.0 interface; the
min-max normalizer maps train actions into [-1, 1] with constant rotation
dims at exactly zero; a short CPU training run halves the denoising MSE and
gets the DDIM-sampled chunk L1 below an absolute normalized bound; DDPM and
DDIM samples are finite; the checkpoint round-trips bitwise (same generator
seed); and the open-loop report over one episode is finite with denormalized
position deltas at the action contract's scale (a min-max wiring invariant —
clip_sample bounds them by the train extrema). No pre-trained checkpoint or
GPU is required::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_diffusion_training.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.dataset import obs_matrix
from alexdoor_xas.policies.common.data import PolicyDataError
from alexdoor_xas.policies.common.inspect import open_loop_report
from alexdoor_xas.policies.diffusion import DiffusionConfigError, load_diffusion_config
from alexdoor_xas.policies.diffusion.checkpoint import load_checkpoint, save_checkpoint
from alexdoor_xas.policies.diffusion.data import (
    MinMaxNormalizer,
    load_diffusion_data,
    make_eval_factory,
    make_train_factory,
)
from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy
from alexdoor_xas.policies.diffusion.schedulers import (
    make_inference_scheduler,
    make_train_scheduler,
    sample_actions,
)
from alexdoor_xas.policies.diffusion.train import (
    make_seeded_model,
    train_diffusion,
)

GATE_OVERRIDES = [
    "model.horizon=8",
    "model.d_model=64",
    "model.n_decoder_layers=2",
    "model.dim_feedforward=128",
    "model.dropout=0.0",
    "model.num_train_timesteps=25",
    "train.epochs=40",
    "train.batch_size=32",
    "train.lr=1e-3",
    "train.lr_schedule=constant",
    "train.lr_warmup_steps=0",
    "train.use_ema=false",
    "train.device=cpu",
    "train.val_every=10",
    "train.val_inference_steps=10",
    "train.overfit_episodes=2",
    "rollout.n_action_steps=4",
    "rollout.num_inference_steps=25",
]
MSE_IMPROVEMENT_FACTOR = 0.5  # untrained epsilon-prediction sits near MSE ~= 1.0
SAMPLED_L1_BOUND = 0.25  # normalized [-1, 1] units; random samples give ~0.5-0.9
POSITION_DELTA_BOUND_M = 0.04  # 2x the frozen 0.02 m per-tick clamp


def check_space(space: str, out_dir: Path, failures: list[str]) -> None:
    label = f"{paths.ALEX_V2_TASK}/{space}"
    try:
        cfg = load_diffusion_config([f"dataset.space={space}", *GATE_OVERRIDES])
    except DiffusionConfigError as error:
        failures.append(f"{label}: config failed to compose: {error}")
        return
    try:
        data = load_diffusion_data(cfg.dataset)
    except PolicyDataError as error:
        failures.append(f"{label}: {error}")
        return

    # Normalizer invariants over the full train split.
    normalizer = MinMaxNormalizer.from_norm_stats(data.stats.action)
    train_actions = np.concatenate(
        [data.dataset.by_id(episode_id).actions for episode_id in data.train_ids]
    )
    normalized = normalizer.normalize(train_actions)
    if not (np.abs(normalized).max() <= 1.0 + 1e-9):
        failures.append(
            f"{label}: train actions escape [-1, 1] after min-max "
            f"(max |a| = {np.abs(normalized).max():.6f})"
        )
    if not (normalized[:, 3:] == 0.0).all():
        failures.append(f"{label}: constant rotation dims did not normalize to exactly 0")

    overfit_ids = data.train_ids[: cfg.train.overfit_episodes]
    make_train = make_train_factory(
        data, cfg.model.horizon, cfg.train.batch_size, cfg.train.seed, episode_ids=overfit_ids
    )
    # Overfit sanity: validate on the same subset the model trains on.
    make_val = make_eval_factory(
        data, cfg.model.horizon, cfg.train.batch_size, cfg.train.seed, overfit_ids
    )

    model = make_seeded_model(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        model_cfg=cfg.model,
        seed=cfg.train.seed,
    )
    scheduler = make_train_scheduler(cfg.model)
    history = train_diffusion(model, scheduler, make_train, cfg.train, make_val_batches=make_val)

    first, last = history.epochs[0], history.epochs[-1]
    if not (last.train_mse < MSE_IMPROVEMENT_FACTOR * first.train_mse):
        failures.append(
            f"{label}: overfit did not halve denoising MSE "
            f"(first {first.train_mse:.4f} -> last {last.train_mse:.4f})"
        )
    if last.val_sampled_l1 is None or not math.isfinite(last.val_sampled_l1):
        failures.append(f"{label}: sampled validation L1 missing or non-finite")
    elif not (last.val_sampled_l1 < SAMPLED_L1_BOUND):
        failures.append(
            f"{label}: sampled chunk L1 {last.val_sampled_l1:.4f} >= bound {SAMPLED_L1_BOUND}"
        )

    # Both samplers produce finite chunks from a real observation.
    model.eval()
    record = data.dataset.by_id(overfit_ids[0])
    obs_row = data.stats.obs.normalize(obs_matrix(record, cfg.dataset.obs_preset)[0])
    obs = torch.as_tensor(obs_row, dtype=torch.float32).reshape(1, -1)
    for sampler_name, steps in (("ddpm", cfg.model.num_train_timesteps), ("ddim", 10)):
        inference = make_inference_scheduler(cfg.model, sampler_name, steps)
        sampled = sample_actions(
            model, inference, obs, cfg.model.horizon, data.action_dim,
            torch.Generator().manual_seed(0),
        )
        if not torch.isfinite(sampled).all():
            failures.append(f"{label}: {sampler_name} sample is non-finite")

    checkpoint_path = out_dir / f"{space}_overfit.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        {"dataset": {"space": space, "obs_preset": cfg.dataset.obs_preset}},
        data.stats,
        meta={"gate": "verify_diffusion_training"},
    )
    loaded = load_checkpoint(checkpoint_path)
    inference = make_inference_scheduler(cfg.model, "ddim", 10)
    original = sample_actions(
        model, inference, obs, cfg.model.horizon, data.action_dim,
        torch.Generator().manual_seed(1),
    )
    inference = make_inference_scheduler(loaded.model.cfg, "ddim", 10)
    rebuilt = sample_actions(
        loaded.model, inference, obs, cfg.model.horizon, data.action_dim,
        torch.Generator().manual_seed(1),
    )
    if not torch.equal(original, rebuilt):
        failures.append(f"{label}: checkpoint round-trip changed predictions")

    policy = DiffusionPolicy.from_checkpoint(
        checkpoint_path, sampler="ddim", num_inference_steps=10
    )
    policy.seed(0)
    report = open_loop_report(
        policy,
        [record],
        json_path=out_dir / f"{space}_open_loop.json",
        stride=cfg.rollout.n_action_steps,
    )
    l1_per_dim = np.asarray(report["aggregate"]["l1_per_dim"])
    if not np.isfinite(l1_per_dim).all():
        failures.append(f"{label}: open-loop report has non-finite errors")
    policy.seed(0)
    chunk = policy.predict(obs_matrix(record, cfg.dataset.obs_preset)[0])
    max_pos_delta = float(np.abs(chunk[:, :3]).max())
    if not (np.isfinite(chunk).all() and max_pos_delta < POSITION_DELTA_BOUND_M):
        failures.append(
            f"{label}: denormalized position deltas out of scale "
            f"(max {max_pos_delta:.4f} m >= {POSITION_DELTA_BOUND_M} m)"
        )

    print(
        f"  [ok ] {label}: denoise MSE {first.train_mse:.4f} -> {last.train_mse:.4f}, "
        f"sampled val L1 {last.val_sampled_l1:.4f}, "
        f"open-loop L1 {report['aggregate']['l1_mean']:.5f}, "
        f"max |dpos| {max_pos_delta * 1000:.2f} mm"
        if not any(failure.startswith(label) for failure in failures)
        else f"  [FAIL] {label}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.OUTPUTS_DIR / "verify_diffusion_training"),
        help="Gate artifact directory.",
    )
    args, _ = parser.parse_known_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for space in (A2_EE_DELTA, A3_OBJ_REL_EE_DELTA):
        check_space(space, out_dir, failures)

    print("-- result --")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("FAIL")
        return 1
    print(
        "PASS (Diffusion Policy training pipeline: A2 + A3 overfit, normalizer "
        "invariants, both samplers, checkpoint, open-loop)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
