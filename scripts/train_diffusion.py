#!/usr/bin/env python
"""Train the Diffusion Policy baseline on one Phase 2 dataset export (Phase 3.3, no Kit).

Consumes episodes only through the frozen Phase 3.0 dataset interface (shared
splits, official norm stats), trains the state-only time-series diffusion
transformer (epsilon prediction, squared-cosine DDPM schedule, EMA weights),
and writes a self-contained run under ``outputs/<experiment>/<run_id>/``:
best/last checkpoints (norm stats embedded; EMA weights when enabled), the
per-epoch loss log, and an open-loop prediction report over the validation
split. Real runs are GPU-backed (``train.device=cuda`` by default); W&B
tracking is optional and disabled by default (no network or login needed)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_diffusion.py \
        dataset.space=A2_ee_delta
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.policies.diffusion import (
    DiffusionConfig,
    DiffusionConfigError,
    load_diffusion_config,
)


def parse_config() -> DiffusionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=str, default=None, help="Action space to train on.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--run-id", type=str, default=None, help="outputs/<experiment>/<run_id>.")
    parser.add_argument("--device", type=str, default=None, help="Training device (cuda/cpu).")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="Restrict training to the first N train-split episodes (sanity mode).",
    )
    args, hydra_overrides = parser.parse_known_args()
    try:
        return load_diffusion_config(
            hydra_overrides,
            cli_overrides={
                "dataset.space": args.space,
                "train.epochs": args.epochs,
                "train.seed": args.seed,
                "train.device": args.device,
                "run.run_id": args.run_id,
                "train.overfit_episodes": args.overfit,
            },
        )
    except DiffusionConfigError as error:
        parser.error(str(error))


def main() -> int:
    cfg = parse_config()

    # Heavy imports after config resolution (config layer stays torch-free).
    import torch

    from alexdoor_xas.policies.common.data import PolicyDataError
    from alexdoor_xas.policies.common.inspect import open_loop_report
    from alexdoor_xas.policies.diffusion.checkpoint import save_checkpoint
    from alexdoor_xas.policies.diffusion.data import (
        load_diffusion_data,
        make_eval_factory,
        make_train_factory,
    )
    from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy
    from alexdoor_xas.policies.diffusion.schedulers import make_train_scheduler
    from alexdoor_xas.policies.diffusion.train import (
        EmaModel,
        make_seeded_model,
        train_diffusion,
    )
    from alexdoor_xas.tracking import load_wandb_config, start_wandb_run

    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "FAIL: train.device=cuda but torch.cuda.is_available() is False — "
            "run on the GPU host or pass train.device=cpu for a smoke run"
        )
        return 1
    device_info = (
        torch.cuda.get_device_name(0)
        if cfg.train.device.startswith("cuda")
        else f"cpu ({torch.get_num_threads()} threads)"
    )

    try:
        data = load_diffusion_data(cfg.dataset)
    except PolicyDataError as error:
        print(f"FAIL: {error}")
        return 1

    run_id = cfg.resolved_run_id()
    output_root = (
        paths.ALEX_V2_OUTPUTS_DIR
        if cfg.dataset.task == paths.ALEX_V2_TASK
        else paths.OUTPUTS_DIR
    )
    run_dir = output_root / cfg.run.experiment / run_id
    checkpoint_dir = run_dir / "checkpoints"
    train_ids = data.train_ids
    if cfg.train.overfit_episodes is not None:
        train_ids = train_ids[: cfg.train.overfit_episodes]

    config_dict = dataclasses.asdict(cfg)
    model = make_seeded_model(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        model_cfg=cfg.model,
        seed=cfg.train.seed,
    )
    scheduler = make_train_scheduler(cfg.model)
    ema = EmaModel(model, cfg.train.ema_decay) if cfg.train.use_ema else None
    print(
        f"[train_diffusion] {cfg.dataset.task}/{cfg.dataset.space}/{cfg.dataset.version} "
        f"obs={cfg.dataset.obs_preset}({data.obs_dim}) action_dim={data.action_dim} "
        f"episodes train={len(train_ids)} val={len(data.val_ids)} "
        f"stats={data.stats_source} params={model.n_parameters:,} "
        f"Tp={cfg.model.horizon} T={cfg.model.num_train_timesteps} "
        f"ema={'on' if ema else 'off'} device={cfg.train.device} ({device_info})"
    )
    if cfg.train.overfit_episodes is not None:
        print(f"[train_diffusion] overfit mode: {len(train_ids)} train episode(s)")

    make_train = make_train_factory(
        data, cfg.model.horizon, cfg.train.batch_size, cfg.train.seed, episode_ids=train_ids
    )
    make_val = make_eval_factory(
        data, cfg.model.horizon, cfg.train.batch_size, cfg.train.seed, data.val_ids
    )

    wandb_cfg = load_wandb_config(
        overrides={
            "group": cfg.run.experiment,
            "name": run_id,
            "job_type": "train",
            **cfg.wandb_overrides,
        }
    )
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    def eval_model():
        return ema.module if ema is not None else model

    wandb_config = {
        **config_dict,
        "n_parameters": model.n_parameters,
        "stats_source": data.stats_source,
        "device_info": device_info,
    }
    with start_wandb_run(wandb_cfg, config=wandb_config) as run:

        def on_epoch(stats, is_best: bool) -> None:
            payload = {
                "epoch": stats.epoch,
                "train/mse": stats.train_mse,
                "train/lr": stats.lr,
            }
            if stats.val_sampled_l1 is not None:
                payload["val/sampled_l1"] = stats.val_sampled_l1
                print(
                    f"[train_diffusion] epoch {stats.epoch + 1}/{cfg.train.epochs} "
                    f"train_mse={stats.train_mse:.5f} "
                    f"val_sampled_l1={stats.val_sampled_l1:.5f}"
                    + (" (best)" if is_best else "")
                )
            run.log(payload)
            meta = {
                "epoch": stats.epoch,
                "val_sampled_l1": stats.val_sampled_l1,
                "run_id": run_id,
                "ema": ema is not None,
                "device_info": device_info,
            }
            if is_best:
                save_checkpoint(
                    best_path,
                    eval_model(),
                    config_dict,
                    data.stats,
                    meta=meta,
                    robot_asset=data.robot_asset,
                    split_episode_ids={
                        "train": train_ids,
                        "val": data.val_ids,
                        "test": data.test_ids,
                    },
                )

        history = train_diffusion(
            model,
            scheduler,
            make_train,
            cfg.train,
            make_val_batches=make_val,
            on_epoch=on_epoch,
            ema=ema,
        )
        save_checkpoint(
            last_path,
            eval_model(),
            config_dict,
            data.stats,
            meta={
                "epoch": cfg.train.epochs - 1,
                "run_id": run_id,
                "ema": ema is not None,
                "device_info": device_info,
            },
            robot_asset=data.robot_asset,
            split_episode_ids={"train": train_ids, "val": data.val_ids, "test": data.test_ids},
        )
        if not best_path.is_file():  # no val improvement recorded (degenerate run)
            save_checkpoint(
                best_path,
                eval_model(),
                config_dict,
                data.stats,
                meta={"run_id": run_id},
                robot_asset=data.robot_asset,
                split_episode_ids={
                    "train": train_ids,
                    "val": data.val_ids,
                    "test": data.test_ids,
                },
            )

        # Open-loop inspection with the DDIM validation sampler (fast, seeded)
        # at the rollout execution stride, so inspection matches deployment.
        policy = DiffusionPolicy.from_checkpoint(
            best_path,
            device="cpu",
            sampler="ddim",
            num_inference_steps=cfg.train.val_inference_steps,
            runtime_asset=data.robot_asset,
        )
        policy.seed(cfg.train.seed)
        val_records = [data.dataset.by_id(episode_id) for episode_id in data.val_ids]
        report = open_loop_report(
            policy,
            val_records,
            json_path=run_dir / "metrics" / "open_loop.json",
            plots_dir=run_dir / "plots",
            stride=cfg.rollout.n_action_steps,
        )
        run.log(
            {
                "open_loop/val_l1_mean": report["aggregate"]["l1_mean"],
                "best/epoch": history.best_epoch,
                "best/val_sampled_l1": history.best_val_l1,
            }
        )

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_log.json").write_text(
        json.dumps(
            {
                "config": _jsonable(config_dict),
                "run_id": run_id,
                "n_parameters": model.n_parameters,
                "stats_source": data.stats_source,
                "robot_asset": data.robot_asset.to_dict() if data.robot_asset else None,
                "device_info": device_info,
                "ema": ema is not None,
                "train_episode_ids": list(train_ids),
                "val_episode_ids": list(data.val_ids),
                "checkpoints": {"best": str(best_path), "last": str(last_path)},
                "history": history.to_dict(),
                "open_loop_val_l1_mean": report["aggregate"]["l1_mean"],
                "open_loop_stride": cfg.rollout.n_action_steps,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"[train_diffusion] done: best epoch {history.best_epoch} "
        f"val_sampled_l1={history.best_val_l1:.5f} "
        f"open-loop val L1={report['aggregate']['l1_mean']:.5f}"
    )
    print(f"[train_diffusion] outputs: {run_dir}")
    return 0


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    sys.exit(main())
