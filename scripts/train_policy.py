#!/usr/bin/env python
"""Train an ACT or Diffusion policy, or resume its incomplete run."""

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
from alexdoor_xas.policies.act.config import (
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
from alexdoor_xas.policies.diffusion.config import (
    DiffusionConfig,
    DiffusionConfigError,
    diffusion_config_from_dict,
    load_diffusion_config,
)

POLICIES = ("act", "diffusion")
LAST_CHECKPOINT_FORMATS = {
    "act": "alexdoor_xas.act.resume.v1",
    "diffusion": "alexdoor_xas.diffusion.resume.v1",
}
PolicyConfig = ActConfig | DiffusionConfig


def parse_config() -> tuple[str, PolicyConfig, Path | None, dict | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        required=True,
        help="Policy family to train; required for new and resumed runs.",
    )
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
        help="Incomplete run directory containing checkpoints/last.pt.",
    )
    args, hydra_overrides = parser.parse_known_args()
    explicit_overrides = {
        "dataset.space": args.space,
        "train.epochs": args.epochs,
        "train.seed": args.seed,
        "train.device": args.device,
        "train.overfit_episodes": args.overfit,
    }
    config_from_dict = act_config_from_dict if args.policy == "act" else diffusion_config_from_dict
    load_config = load_act_config if args.policy == "act" else load_diffusion_config

    if args.resume is not None:
        if hydra_overrides or any(value is not None for value in explicit_overrides.values()):
            parser.error("--resume loads the frozen config and cannot be combined with overrides")
        try:
            run_dir = resolve_resume_directory(args.resume, args.policy)
            resolved = load_resolved_config(run_dir)
            if resolved.get("run_type") != "training" or resolved.get("policy") != args.policy:
                raise ValueError(f"--resume must name a {args.policy.upper()} training run")
            return args.policy, config_from_dict(resolved["config"]), run_dir, resolved
        except (ActConfigError, DiffusionConfigError, KeyError, ValueError) as error:
            parser.error(str(error))

    try:
        cfg = load_config(hydra_overrides, cli_overrides=explicit_overrides)
    except (ActConfigError, DiffusionConfigError) as error:
        parser.error(str(error))
    return args.policy, cfg, None, None


def main() -> int:
    policy_name, cfg, resume_dir, resolved = parse_config()

    import torch

    from alexdoor_xas.policies.act.checkpoint import save_checkpoint as save_act_checkpoint
    from alexdoor_xas.policies.act.policy import ActPolicy
    from alexdoor_xas.policies.act.train import (
        make_seeded_model as make_seeded_act_model,
    )
    from alexdoor_xas.policies.act.train import train_act
    from alexdoor_xas.policies.common.data import (
        PolicyDataError,
        load_policy_data,
    )
    from alexdoor_xas.policies.common.data import (
        make_eval_factory as make_act_eval_factory,
    )
    from alexdoor_xas.policies.common.data import make_train_factory as make_act_train_factory
    from alexdoor_xas.policies.common.inspect import open_loop_report
    from alexdoor_xas.policies.diffusion.checkpoint import (
        save_checkpoint as save_diffusion_checkpoint,
    )
    from alexdoor_xas.policies.diffusion.data import (
        load_diffusion_data,
    )
    from alexdoor_xas.policies.diffusion.data import (
        make_eval_factory as make_diffusion_eval_factory,
    )
    from alexdoor_xas.policies.diffusion.data import (
        make_train_factory as make_diffusion_train_factory,
    )
    from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy
    from alexdoor_xas.policies.diffusion.schedulers import make_train_scheduler
    from alexdoor_xas.policies.diffusion.train import (
        EmaModel,
        train_diffusion,
    )
    from alexdoor_xas.policies.diffusion.train import (
        make_seeded_model as make_seeded_diffusion_model,
    )

    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("FAIL: train.device requests CUDA but no CUDA device is visible")
        return 2

    try:
        data = (
            load_policy_data(cfg.dataset)
            if policy_name == "act"
            else load_diffusion_data(cfg.dataset)
        )
    except PolicyDataError as error:
        print(f"FAIL: {error}")
        return 1

    if resume_dir is None:
        run_id, run_dir = allocate_run_directory(
            output_root=cfg.run.output_root,
            policy=policy_name,
            action_space=cfg.dataset.space,
            dataset_version=cfg.dataset.version,
            dataset_view_id=cfg.dataset.view_id,
            seed=cfg.train.seed,
        )
        resolved = resolved_training_config(
            run_id=run_id,
            policy=policy_name,
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
    if policy_name == "act":
        model = make_seeded_act_model(data.obs_dim, data.action_dim, cfg.model, cfg.train.seed)
        horizon = cfg.model.chunk_size
        make_train = make_act_train_factory(
            data,
            horizon,
            cfg.train.batch_size,
            cfg.train.seed,
            episode_ids=train_ids,
        )
        make_val = make_act_eval_factory(
            data, horizon, cfg.train.batch_size, cfg.train.seed, data.val_ids
        )
        noise_scheduler = None
        ema = None
    else:
        model = make_seeded_diffusion_model(
            data.obs_dim, data.action_dim, cfg.model, cfg.train.seed
        )
        horizon = cfg.model.horizon
        make_train = make_diffusion_train_factory(
            data,
            horizon,
            cfg.train.batch_size,
            cfg.train.seed,
            episode_ids=train_ids,
        )
        make_val = make_diffusion_eval_factory(
            data, horizon, cfg.train.batch_size, cfg.train.seed, data.val_ids
        )
        noise_scheduler = make_train_scheduler(cfg.model)
        ema = EmaModel(model, cfg.train.ema_decay) if cfg.train.use_ema else None

    resume_state = None
    if resume_dir is not None:
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_last_checkpoint(resume_payload, run_id, policy_name)
        model.load_state_dict(resume_payload["model_state"])
        resume_state = resume_payload["training_state"]

    device_info = (
        torch.cuda.get_device_name(torch.device(cfg.train.device))
        if cfg.train.device.startswith("cuda")
        else "CPU"
    )
    print(f"[train_policy:{policy_name}] run={run_id} device={cfg.train.device} ({device_info})")

    def eval_model():
        return ema.module if ema is not None else model

    def save_last(training_state) -> None:
        torch_save_atomic(
            last_path,
            {
                "format": LAST_CHECKPOINT_FORMATS[policy_name],
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
            os.environ.setdefault("WANDB_RUN_GROUP", policy_name)
            os.environ.setdefault("WANDB_JOB_TYPE", "train")
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError('W&B tracking requires: pip install -e ".[tracking]"') from error
            tracking = wandb.init(
                dir=str(paths.OUTPUTS_DIR),
                config={
                    "policy": policy_name,
                    "run_id": run_id,
                    "dataset": resolved["config"]["dataset"],
                    "model": resolved["config"]["model"],
                    "train": resolved["config"]["train"],
                },
            )

        with tracking as run:

            def on_epoch(stats, is_best: bool) -> None:
                if run is not None:
                    run.log({"epoch": stats.epoch, **_epoch_metrics(policy_name, stats)})
                if is_best:
                    meta = _best_checkpoint_meta(policy_name, stats, run_id, ema is not None)
                    if policy_name == "act":
                        save_act_checkpoint(
                            best_path,
                            model,
                            resolved["config"],
                            data.stats,
                            meta=meta,
                            robot_asset=data.robot_asset,
                        )
                    else:
                        save_diffusion_checkpoint(
                            best_path,
                            eval_model(),
                            resolved["config"],
                            data.stats,
                            meta=meta,
                            robot_asset=data.robot_asset,
                        )

            if policy_name == "act":
                history = train_act(
                    model,
                    make_train,
                    cfg.train,
                    make_val_batches=make_val,
                    on_epoch=on_epoch,
                    resume_state=resume_state,
                    on_checkpoint=save_last,
                )
            else:
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
                meta = {"run_id": run_id}
                if policy_name == "act":
                    save_act_checkpoint(
                        best_path,
                        model,
                        resolved["config"],
                        data.stats,
                        meta=meta,
                        robot_asset=data.robot_asset,
                    )
                else:
                    meta["ema"] = ema is not None
                    save_diffusion_checkpoint(
                        best_path,
                        eval_model(),
                        resolved["config"],
                        data.stats,
                        meta=meta,
                        robot_asset=data.robot_asset,
                    )

            history_payload = history.to_dict()
            write_json_atomic(run_dir / "training" / "history.json", history_payload)
            write_training_summary(
                policy_name, history_payload, run_dir / "training" / "summary.png"
            )
            if policy_name == "act":
                policy = ActPolicy.from_checkpoint(
                    best_path,
                    device=cfg.train.device,
                    runtime_asset=data.robot_asset,
                )
                open_loop_stride = 1
            else:
                policy = DiffusionPolicy.from_checkpoint(
                    best_path,
                    device=cfg.train.device,
                    sampler="ddim",
                    num_inference_steps=cfg.train.val_inference_steps,
                    runtime_asset=data.robot_asset,
                )
                policy.seed(cfg.train.seed)
                open_loop_stride = cfg.rollout.n_action_steps

            val_records = [data.dataset.by_id(episode_id) for episode_id in data.val_ids]
            open_loop = open_loop_report(
                policy,
                val_records,
                json_path=run_dir / "open_loop" / "metrics.json",
                summary_path=run_dir / "open_loop" / "summary.png",
                stride=open_loop_stride,
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

    print(f"[train_policy:{policy_name}] completed: {run_dir}")
    return 0


def _epoch_metrics(policy_name: str, stats) -> dict[str, float]:
    if policy_name == "act":
        return {
            "train/l1": stats.train_l1,
            "train/kl": stats.train_kl,
            "train/loss": stats.train_loss,
            "val/l1": stats.val_l1,
        }
    return {
        "train/mse": stats.train_mse,
        "train/lr": stats.lr,
        "val/sampled_l1": stats.val_sampled_l1,
    }


def _best_checkpoint_meta(policy_name: str, stats, run_id: str, using_ema: bool) -> dict:
    if policy_name == "act":
        return {"epoch": stats.epoch, "val_l1": stats.val_l1, "run_id": run_id}
    return {
        "epoch": stats.epoch,
        "val_sampled_l1": stats.val_sampled_l1,
        "run_id": run_id,
        "ema": using_ema,
    }


def _validate_last_checkpoint(payload: dict, run_id: str, policy_name: str) -> None:
    if (
        payload.get("format") != LAST_CHECKPOINT_FORMATS[policy_name]
        or payload.get("run_id") != run_id
    ):
        raise ValueError(f"last.pt does not match the requested {policy_name.upper()} run")


def _record_error(run_dir: Path, error: BaseException) -> None:
    target = run_dir / "error.log"
    previous = target.read_text() if target.is_file() else ""
    entry = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    target.write_text(previous + ("\n" if previous else "") + entry)


if __name__ == "__main__":
    sys.exit(main())
