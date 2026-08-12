#!/usr/bin/env python
"""Train ACT into one exclusive canonical run, or resume an incomplete run."""

from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.policies.act import (
    ActConfig,
    ActConfigError,
    act_config_from_dict,
    load_act_config,
)
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

LAST_CHECKPOINT_FORMAT = "alexdoor_xas.act.resume.v1"


def parse_config() -> tuple[ActConfig, Path | None, dict | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=str, default=None, help="Action space to train on.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
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
        help="Explicit incomplete ACT run directory containing checkpoints/last.pt.",
    )
    args, hydra_overrides = parser.parse_known_args()
    explicit_overrides = {
        "dataset.space": args.space,
        "train.epochs": args.epochs,
        "train.seed": args.seed,
        "train.overfit_episodes": args.overfit,
    }
    if args.resume is not None:
        if hydra_overrides or any(value is not None for value in explicit_overrides.values()):
            parser.error("--resume loads the frozen config and cannot be combined with overrides")
        try:
            run_dir = resolve_resume_directory(args.resume, "act")
            resolved = load_resolved_config(run_dir)
            if resolved.get("run_type") != "training" or resolved.get("policy") != "act":
                raise ValueError("--resume must name an ACT training run")
            return act_config_from_dict(resolved["config"]), run_dir, resolved
        except (ActConfigError, KeyError, ValueError) as error:
            parser.error(str(error))
    try:
        return (
            load_act_config(hydra_overrides, cli_overrides=explicit_overrides),
            None,
            None,
        )
    except ActConfigError as error:
        parser.error(str(error))


def main() -> int:
    cfg, resume_dir, resolved = parse_config()

    import torch

    from alexdoor_xas.policies.act.checkpoint import save_checkpoint
    from alexdoor_xas.policies.act.data import (
        ActDataError,
        load_act_data,
        make_eval_factory,
        make_train_factory,
    )
    from alexdoor_xas.policies.act.policy import ActPolicy
    from alexdoor_xas.policies.act.train import make_seeded_model, train_act
    from alexdoor_xas.policies.common.inspect import open_loop_report

    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("FAIL: train.device requests CUDA but no CUDA device is visible")
        return 2
    try:
        data = load_act_data(cfg.dataset)
    except ActDataError as error:
        print(f"FAIL: {error}")
        return 1

    if resume_dir is None:
        run_id, run_dir = allocate_run_directory(
            output_root=cfg.run.output_root,
            policy="act",
            action_space=cfg.dataset.space,
            dataset_version=cfg.dataset.version,
            dataset_view_id=cfg.dataset.view_id,
            seed=cfg.train.seed,
        )
        resolved = resolved_training_config(
            run_id=run_id,
            policy="act",
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
    resume_state = None
    if resume_dir is not None:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_last_checkpoint(resume_payload, run_id)
        model.load_state_dict(resume_payload["model_state"])
        resume_state = resume_payload["training_state"]

    make_train = make_train_factory(
        data,
        cfg.model.chunk_size,
        cfg.train.batch_size,
        cfg.train.seed,
        episode_ids=train_ids,
    )
    make_val = make_eval_factory(
        data, cfg.model.chunk_size, cfg.train.batch_size, cfg.train.seed, data.val_ids
    )
    device_info = (
        torch.cuda.get_device_name(torch.device(cfg.train.device))
        if cfg.train.device.startswith("cuda")
        else "CPU"
    )
    print(f"[train_act] run={run_id} device={cfg.train.device} ({device_info})")

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
        tracking = nullcontext(None)
        if os.environ.get("WANDB_MODE", "disabled").strip().lower() != "disabled":
            os.environ.setdefault("WANDB_PROJECT", "alexdoor-xas")
            os.environ.setdefault("WANDB_NAME", run_id)
            os.environ.setdefault("WANDB_RUN_GROUP", "act")
            os.environ.setdefault("WANDB_JOB_TYPE", "train")
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    'W&B tracking requires: pip install -e ".[tracking]"'
                ) from error
            tracking = wandb.init(
                dir=str(paths.OUTPUTS_DIR),
                config={
                    "policy": "act",
                    "run_id": run_id,
                    "dataset": resolved["config"]["dataset"],
                    "model": resolved["config"]["model"],
                    "train": resolved["config"]["train"],
                },
            )

        with tracking as run:

            def on_epoch(stats, is_best: bool) -> None:
                if run is not None:
                    run.log(
                        {
                            "epoch": stats.epoch,
                            "train/l1": stats.train_l1,
                            "train/kl": stats.train_kl,
                            "train/loss": stats.train_loss,
                            "val/l1": stats.val_l1,
                        }
                    )
                if is_best:
                    save_checkpoint(
                        best_path,
                        model,
                        resolved["config"],
                        data.stats,
                        meta={"epoch": stats.epoch, "val_l1": stats.val_l1, "run_id": run_id},
                        robot_asset=data.robot_asset,
                    )

            history = train_act(
                model,
                make_train,
                cfg.train,
                make_val_batches=make_val,
                on_epoch=on_epoch,
                resume_state=resume_state,
                on_checkpoint=save_last,
            )
            if not best_path.is_file():
                save_checkpoint(
                    best_path,
                    model,
                    resolved["config"],
                    data.stats,
                    meta={"run_id": run_id},
                    robot_asset=data.robot_asset,
                )

            history_payload = history.to_dict()
            write_json_atomic(run_dir / "training" / "history.json", history_payload)
            write_training_summary("act", history_payload, run_dir / "training" / "summary.png")
            policy = ActPolicy.from_checkpoint(
                best_path,
                device=cfg.train.device,
                runtime_asset=data.robot_asset,
            )
            val_records = [data.dataset.by_id(episode_id) for episode_id in data.val_ids]
            open_loop = open_loop_report(
                policy,
                val_records,
                json_path=run_dir / "open_loop" / "metrics.json",
                summary_path=run_dir / "open_loop" / "summary.png",
            )
            if run is not None:
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

    print(f"[train_act] completed: {run_dir}")
    return 0


def _validate_last_checkpoint(payload: dict, run_id: str) -> None:
    if payload.get("format") != LAST_CHECKPOINT_FORMAT or payload.get("run_id") != run_id:
        raise ValueError("last.pt does not match the requested ACT run")


def _record_error(run_dir: Path, error: BaseException) -> None:
    target = run_dir / "error.log"
    previous = target.read_text() if target.is_file() else ""
    entry = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    target.write_text(previous + ("\n" if previous else "") + entry)


if __name__ == "__main__":
    sys.exit(main())
