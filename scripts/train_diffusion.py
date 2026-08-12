#!/usr/bin/env python
"""Train Diffusion Policy into one exclusive run, or resume an incomplete run."""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np

from alexdoor_xas.policies.common.runs import (
    allocate_run_directory,
    load_resolved_config,
    resolve_resume_directory,
    resolved_training_config,
    torch_save_atomic,
    write_json_atomic,
    write_run_report,
    write_training_summary,
)
from alexdoor_xas.policies.diffusion import (
    DiffusionConfig,
    DiffusionConfigError,
    diffusion_config_from_dict,
    load_diffusion_config,
)

LAST_CHECKPOINT_FORMAT = "alexdoor_xas.diffusion.resume.v1"


def parse_config() -> tuple[DiffusionConfig, Path | None, dict | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=str, default=None, help="Action space to train on.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--device", type=str, default=None, help="Training device.")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="Restrict training to the first N train-split episodes.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Explicit incomplete Diffusion run directory containing checkpoints/last.pt.",
    )
    args, hydra_overrides = parser.parse_known_args()
    explicit_overrides = {
        "dataset.space": args.space,
        "train.epochs": args.epochs,
        "train.seed": args.seed,
        "train.device": args.device,
        "train.overfit_episodes": args.overfit,
    }
    if args.resume is not None:
        if hydra_overrides or any(value is not None for value in explicit_overrides.values()):
            parser.error("--resume loads the frozen config and cannot be combined with overrides")
        try:
            run_dir = resolve_resume_directory(args.resume, "diffusion")
            resolved = load_resolved_config(run_dir)
            if resolved.get("run_type") != "training" or resolved.get("policy") != "diffusion":
                raise ValueError("--resume must name a Diffusion training run")
            return diffusion_config_from_dict(resolved["config"]), run_dir, resolved
        except (DiffusionConfigError, KeyError, ValueError) as error:
            parser.error(str(error))
    try:
        return (
            load_diffusion_config(hydra_overrides, cli_overrides=explicit_overrides),
            None,
            None,
        )
    except DiffusionConfigError as error:
        parser.error(str(error))


def main() -> int:
    cfg, resume_dir, resolved = parse_config()

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
    from alexdoor_xas.policies.diffusion.train import EmaModel, make_seeded_model, train_diffusion
    from alexdoor_xas.tracking import load_wandb_config, start_wandb_run

    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("FAIL: train.device requests CUDA but no CUDA device is visible")
        return 2
    try:
        data = load_diffusion_data(cfg.dataset)
    except PolicyDataError as error:
        print(f"FAIL: {error}")
        return 1

    if resume_dir is None:
        run_id, run_dir = allocate_run_directory(
            output_root=cfg.run.output_root,
            policy="diffusion",
            action_space=cfg.dataset.space,
            dataset_version=cfg.dataset.version,
            dataset_view_id=cfg.dataset.view_id,
            seed=cfg.train.seed,
        )
        resolved = resolved_training_config(
            run_id=run_id,
            policy="diffusion",
            config=cfg,
        )
        write_json_atomic(run_dir / "resolved_config.json", resolved, exclusive=True)
    else:
        run_dir = resume_dir
        run_id = str(resolved["run_id"])

    checkpoint_dir = run_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    train_ids = data.train_ids
    if cfg.train.overfit_episodes is not None:
        train_ids = train_ids[: cfg.train.overfit_episodes]

    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    model = make_seeded_model(data.obs_dim, data.action_dim, cfg.model, cfg.train.seed)
    noise_scheduler = make_train_scheduler(cfg.model)
    ema = EmaModel(model, cfg.train.ema_decay) if cfg.train.use_ema else None
    resume_state = None
    if resume_dir is not None:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_last_checkpoint(resume_payload, run_id)
        model.load_state_dict(resume_payload["model_state"])
        resume_state = resume_payload["training_state"]

    make_train = make_train_factory(
        data,
        cfg.model.horizon,
        cfg.train.batch_size,
        cfg.train.seed,
        episode_ids=train_ids,
    )
    make_val = make_eval_factory(
        data, cfg.model.horizon, cfg.train.batch_size, cfg.train.seed, data.val_ids
    )
    device_info = (
        torch.cuda.get_device_name(torch.device(cfg.train.device))
        if cfg.train.device.startswith("cuda")
        else "CPU"
    )
    print(f"[train_diffusion] run={run_id} device={cfg.train.device} ({device_info})")

    def eval_model():
        return ema.module if ema is not None else model

    def save_last(training_state) -> None:
        torch_save_atomic(
            last_path,
            {
                "format": LAST_CHECKPOINT_FORMAT,
                "run_id": run_id,
                "model_state": model.state_dict(),
                "training_state": training_state,
            },
        )

    try:
        wandb_cfg = load_wandb_config(
            overrides={
                "group": "diffusion",
                "name": run_id,
                "job_type": "train",
                **cfg.wandb_overrides,
            }
        )
        with start_wandb_run(wandb_cfg, config=resolved["config"]) as run:

            def on_epoch(stats, is_best: bool) -> None:
                run.log(
                    {
                        "epoch": stats.epoch,
                        "train/mse": stats.train_mse,
                        "train/lr": stats.lr,
                        "val/sampled_l1": stats.val_sampled_l1,
                    }
                )
                if is_best:
                    save_checkpoint(
                        best_path,
                        eval_model(),
                        resolved["config"],
                        data.stats,
                        meta={
                            "epoch": stats.epoch,
                            "val_sampled_l1": stats.val_sampled_l1,
                            "run_id": run_id,
                            "ema": ema is not None,
                        },
                        robot_asset=data.robot_asset,
                    )

            history = train_diffusion(
                model,
                noise_scheduler,
                make_train,
                cfg.train,
                make_val_batches=make_val,
                on_epoch=on_epoch,
                ema=ema,
                resume_state=resume_state,
                on_checkpoint=save_last,
            )
            if not best_path.is_file():
                save_checkpoint(
                    best_path,
                    eval_model(),
                    resolved["config"],
                    data.stats,
                    meta={"run_id": run_id, "ema": ema is not None},
                    robot_asset=data.robot_asset,
                )

            history_payload = history.to_dict()
            write_json_atomic(run_dir / "training" / "history.json", history_payload)
            write_training_summary(
                "diffusion", history_payload, run_dir / "training" / "summary.png"
            )
            policy = DiffusionPolicy.from_checkpoint(
                best_path,
                device=cfg.train.device,
                sampler="ddim",
                num_inference_steps=cfg.train.val_inference_steps,
                runtime_asset=data.robot_asset,
            )
            policy.seed(cfg.train.seed)
            val_records = [data.dataset.by_id(episode_id) for episode_id in data.val_ids]
            open_loop = open_loop_report(
                policy,
                val_records,
                json_path=run_dir / "open_loop" / "metrics.json",
                summary_path=run_dir / "open_loop" / "summary.png",
                stride=cfg.rollout.n_action_steps,
            )
            run.log(
                {
                    "open_loop/l1_mean": open_loop["aggregate_l1_mean"],
                    "best/epoch": history.best_epoch,
                    "best/value": history.best_val_l1,
                }
            )

        write_run_report(
            run_dir,
            resolved,
            status="completed",
            training={
                "best_epoch": history.best_epoch,
                "best_value": history.best_val_l1,
                "duration_s": history.duration_s,
            },
            open_loop=open_loop,
        )
        last_path.unlink()
    except BaseException as error:  # retain resumable state for errors and interruptions
        _record_error(run_dir, error)
        write_run_report(
            run_dir,
            resolved,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            anomalies=[f"{type(error).__name__}: {error}"],
        )
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1

    print(f"[train_diffusion] completed: {run_dir}")
    return 0


def _validate_last_checkpoint(payload: dict, run_id: str) -> None:
    if payload.get("format") != LAST_CHECKPOINT_FORMAT or payload.get("run_id") != run_id:
        raise ValueError("last.pt does not match the requested Diffusion run")


def _record_error(run_dir: Path, error: BaseException) -> None:
    target = run_dir / "error.log"
    previous = target.read_text() if target.is_file() else ""
    entry = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    target.write_text(previous + ("\n" if previous else "") + entry)


if __name__ == "__main__":
    sys.exit(main())
