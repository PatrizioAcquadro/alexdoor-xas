#!/usr/bin/env python
"""Train the ACT baseline on one Phase 2 dataset export (Phase 3.2, no Kit).

Consumes episodes only through the frozen Phase 3.0 dataset interface (shared
splits, official norm stats), trains the state-only ACT CVAE, and writes a
self-contained run under ``outputs/<experiment>/<run_id>/``: best/last
checkpoints (norm stats embedded), the per-epoch loss log, and an open-loop
prediction report over the validation split. W&B tracking is optional and
disabled by default (no network or login needed)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py \
        dataset.space=A2_ee_delta
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.policies.act import ActConfig, ActConfigError, load_act_config


def parse_config() -> ActConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=str, default=None, help="Action space to train on.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--run-id", type=str, default=None, help="outputs/<experiment>/<run_id>.")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="Restrict training to the first N train-split episodes (sanity mode).",
    )
    args, hydra_overrides = parser.parse_known_args()
    try:
        return load_act_config(
            hydra_overrides,
            cli_overrides={
                "dataset.space": args.space,
                "train.epochs": args.epochs,
                "train.seed": args.seed,
                "run.run_id": args.run_id,
                "train.overfit_episodes": args.overfit,
            },
        )
    except ActConfigError as error:
        parser.error(str(error))


def main() -> int:
    cfg = parse_config()

    # Heavy imports after config resolution (config layer stays torch-free).
    from alexdoor_xas.policies.act.checkpoint import save_checkpoint
    from alexdoor_xas.policies.act.data import (
        ActDataError,
        load_act_data,
        make_eval_factory,
        make_train_factory,
    )
    from alexdoor_xas.policies.act.inspect import open_loop_report
    from alexdoor_xas.policies.act.model import ACTModel
    from alexdoor_xas.policies.act.policy import ActPolicy
    from alexdoor_xas.policies.act.train import train_act
    from alexdoor_xas.tracking import load_wandb_config, start_wandb_run

    try:
        data = load_act_data(cfg.dataset)
    except ActDataError as error:
        print(f"FAIL: {error}")
        return 1

    run_id = cfg.resolved_run_id()
    run_dir = paths.OUTPUTS_DIR / cfg.run.experiment / run_id
    checkpoint_dir = run_dir / "checkpoints"
    train_ids = data.train_ids
    if cfg.train.overfit_episodes is not None:
        train_ids = train_ids[: cfg.train.overfit_episodes]

    config_dict = dataclasses.asdict(cfg)
    model = ACTModel(obs_dim=data.obs_dim, action_dim=data.action_dim, cfg=cfg.model)
    print(
        f"[train_act] {cfg.dataset.task}/{cfg.dataset.space}/{cfg.dataset.version} "
        f"obs={cfg.dataset.obs_preset}({data.obs_dim}) action_dim={data.action_dim} "
        f"episodes train={len(train_ids)} val={len(data.val_ids)} "
        f"stats={data.stats_source} params={model.n_parameters:,}"
    )
    if cfg.train.overfit_episodes is not None:
        print(f"[train_act] overfit mode: {len(train_ids)} train episode(s)")

    make_train = make_train_factory(
        data, cfg.model.chunk_size, cfg.train.batch_size, cfg.train.seed, episode_ids=train_ids
    )
    make_val = make_eval_factory(
        data, cfg.model.chunk_size, cfg.train.batch_size, cfg.train.seed, data.val_ids
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

    with start_wandb_run(wandb_cfg, config=config_dict) as run:

        def on_epoch(stats, is_best: bool) -> None:
            payload = {
                "epoch": stats.epoch,
                "train/l1": stats.train_l1,
                "train/kl": stats.train_kl,
                "train/loss": stats.train_loss,
            }
            if stats.val_l1 is not None:
                payload["val/l1"] = stats.val_l1
                print(
                    f"[train_act] epoch {stats.epoch + 1}/{cfg.train.epochs} "
                    f"train_l1={stats.train_l1:.5f} val_l1={stats.val_l1:.5f}"
                    + (" (best)" if is_best else "")
                )
            run.log(payload)
            meta = {"epoch": stats.epoch, "val_l1": stats.val_l1, "run_id": run_id}
            if is_best:
                save_checkpoint(best_path, model, config_dict, data.stats, meta=meta)

        history = train_act(
            model, make_train, cfg.train, make_val_batches=make_val, on_epoch=on_epoch
        )
        save_checkpoint(
            last_path,
            model,
            config_dict,
            data.stats,
            meta={"epoch": cfg.train.epochs - 1, "run_id": run_id},
        )
        if not best_path.is_file():  # no val improvement recorded (degenerate run)
            save_checkpoint(best_path, model, config_dict, data.stats, meta={"run_id": run_id})

        policy = ActPolicy.from_checkpoint(best_path, device=cfg.train.device)
        val_records = [data.dataset.by_id(episode_id) for episode_id in data.val_ids]
        report = open_loop_report(
            policy,
            val_records,
            json_path=run_dir / "metrics" / "open_loop.json",
            plots_dir=run_dir / "plots",
        )
        run.log(
            {
                "open_loop/val_l1_mean": report["aggregate"]["l1_mean"],
                "best/epoch": history.best_epoch,
                "best/val_l1": history.best_val_l1,
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
                "train_episode_ids": list(train_ids),
                "val_episode_ids": list(data.val_ids),
                "checkpoints": {"best": str(best_path), "last": str(last_path)},
                "history": history.to_dict(),
                "open_loop_val_l1_mean": report["aggregate"]["l1_mean"],
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"[train_act] done: best epoch {history.best_epoch} "
        f"val_l1={history.best_val_l1:.5f} "
        f"open-loop val L1={report['aggregate']['l1_mean']:.5f}"
    )
    print(f"[train_act] outputs: {run_dir}")
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
