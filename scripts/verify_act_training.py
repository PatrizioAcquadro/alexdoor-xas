#!/usr/bin/env python
"""Phase 3.2 training gate: ACT learns the exported Phase 2 data (no Kit).

For each trainable action space (A2, A3) on the Alex task this verifies, with
a small model and a 2-episode overfit subset: the Hydra config composes; the
dataset/splits/norm-stats triple loads and validates through the Phase 3.0
interface; a short training run cuts the train L1 by at least half (and below
an absolute normalized bound); the checkpoint round-trips bitwise; and the
open-loop report over one episode is finite with denormalized position deltas
at the action contract's scale. No pre-trained checkpoint is required::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_training.py
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
from alexdoor_xas.policies.act import ActConfigError, load_act_config
from alexdoor_xas.policies.act.checkpoint import load_checkpoint, save_checkpoint
from alexdoor_xas.policies.act.data import (
    ActDataError,
    load_act_data,
    make_eval_factory,
    make_train_factory,
)
from alexdoor_xas.policies.act.inspect import open_loop_report
from alexdoor_xas.policies.act.policy import ActPolicy
from alexdoor_xas.policies.act.train import make_seeded_model, train_act

GATE_OVERRIDES = [
    "model.chunk_size=20",
    "model.d_model=64",
    "model.dim_feedforward=128",
    "model.z_dim=8",
    "model.cvae_encoder_layers=1",
    "model.encoder_layers=1",
    "model.decoder_layers=1",
    "model.dropout=0.0",
    "train.epochs=15",
    "train.batch_size=32",
    "train.lr=1e-3",
    "train.kl_weight=1.0",
    "train.val_every=5",
    "train.overfit_episodes=2",
    # The gate is a tiny CPU smoke by design; the GPU-first production default
    # would otherwise leave the gate's direct model calls on mixed devices
    # (same pin the diffusion training gate carries).
    "train.device=cpu",
]
L1_IMPROVEMENT_FACTOR = 0.5
L1_ABSOLUTE_BOUND = 0.15  # normalized units
POSITION_DELTA_BOUND_M = 0.04  # 2x the frozen 0.02 m per-tick clamp


def check_space(space: str, out_dir, failures: list[str]) -> None:
    label = f"{paths.ALEX_V2_TASK}/{space}"
    try:
        cfg = load_act_config([f"dataset.space={space}", *GATE_OVERRIDES])
    except ActConfigError as error:
        failures.append(f"{label}: config failed to compose: {error}")
        return
    try:
        data = load_act_data(cfg.dataset)
    except ActDataError as error:
        failures.append(f"{label}: {error}")
        return

    overfit_ids = data.train_ids[: cfg.train.overfit_episodes]
    make_train = make_train_factory(
        data, cfg.model.chunk_size, cfg.train.batch_size, cfg.train.seed, episode_ids=overfit_ids
    )
    # Overfit sanity: validate on the same subset the model trains on.
    make_val = make_eval_factory(
        data, cfg.model.chunk_size, cfg.train.batch_size, cfg.train.seed, overfit_ids
    )

    model = make_seeded_model(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        model_cfg=cfg.model,
        seed=cfg.train.seed,
    )
    history = train_act(model, make_train, cfg.train, make_val_batches=make_val)

    first, last = history.epochs[0], history.epochs[-1]
    if not (last.train_l1 < L1_IMPROVEMENT_FACTOR * first.train_l1):
        failures.append(
            f"{label}: overfit did not halve train L1 "
            f"(first {first.train_l1:.4f} -> last {last.train_l1:.4f})"
        )
    if not (last.train_l1 < L1_ABSOLUTE_BOUND):
        failures.append(
            f"{label}: overfit train L1 {last.train_l1:.4f} >= bound {L1_ABSOLUTE_BOUND}"
        )
    if last.val_l1 is None or not math.isfinite(last.val_l1):
        failures.append(f"{label}: overfit validation L1 missing or non-finite")

    checkpoint_path = out_dir / f"{space}_overfit.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        {"dataset": {"space": space, "obs_preset": cfg.dataset.obs_preset}},
        data.stats,
        meta={"gate": "verify_act_training"},
    )
    loaded = load_checkpoint(checkpoint_path)
    probe = torch.zeros(1, data.obs_dim)
    model.eval()
    if not torch.equal(loaded.model.predict(probe), model.predict(probe)):
        failures.append(f"{label}: checkpoint round-trip changed predictions")

    policy = ActPolicy.from_checkpoint(checkpoint_path)
    record = data.dataset.by_id(overfit_ids[0])
    report = open_loop_report(
        policy, [record], json_path=out_dir / f"{space}_open_loop.json"
    )
    l1_per_dim = np.asarray(report["aggregate"]["l1_per_dim"])
    if not np.isfinite(l1_per_dim).all():
        failures.append(f"{label}: open-loop report has non-finite errors")
    chunk = policy.predict(np.zeros(data.obs_dim))  # any in-range obs
    max_pos_delta = float(np.abs(chunk[:, :3]).max())
    if not (np.isfinite(chunk).all() and max_pos_delta < POSITION_DELTA_BOUND_M):
        failures.append(
            f"{label}: denormalized position deltas out of scale "
            f"(max {max_pos_delta:.4f} m >= {POSITION_DELTA_BOUND_M} m)"
        )

    print(
        f"  [ok ] {label}: train L1 {first.train_l1:.4f} -> {last.train_l1:.4f}, "
        f"val L1 {last.val_l1:.4f}, open-loop L1 {report['aggregate']['l1_mean']:.5f}, "
        f"max |dpos| {max_pos_delta * 1000:.2f} mm"
        if not any(failure.startswith(label) for failure in failures)
        else f"  [FAIL] {label}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.OUTPUTS_DIR / "verify_act_training"),
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
    print("PASS (ACT training pipeline: A2 + A3 overfit, checkpoint, open-loop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
