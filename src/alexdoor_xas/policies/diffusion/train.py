"""Denoising training loop (torch only; no Isaac imports, no file I/O).

Mirrors the ACT trainer: the loop consumes *batch factories* producing
normalized numpy batch dicts (``obs``, ``actions``, ``is_pad``), so it is
unit-testable with hand-built batches and the script layer owns
dataset/paths/tracking concerns. Two diffusion-specific pieces: an EMA shadow
of the weights (the deployed policy, per the paper) and a sampled-chunk L1
validation metric — the denoising MSE alone is a noisy selector and is not
the quantity that matters closed-loop.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from alexdoor_xas.dataset import collate_torch
from alexdoor_xas.policies.common.runs import capture_rng_states, restore_rng_states
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg, DiffusionTrainCfg
from alexdoor_xas.policies.diffusion.model import DiffusionTransformer, diffusion_loss
from alexdoor_xas.policies.diffusion.schedulers import (
    make_inference_scheduler,
    sample_actions,
)

TrainBatchFactory = Callable[[int], Iterable[dict]]
"""``epoch -> iterable of normalized numpy batch dicts`` (fresh shuffle per epoch)."""
ValBatchFactory = Callable[[], Iterable[dict]]

VAL_SAMPLING_SEED = 12345
"""Fixed generator seed for the sampled validation metric, so best-checkpoint
selection compares epochs on identical noise draws."""


def make_seeded_model(
    obs_dim: int,
    action_dim: int,
    model_cfg: DiffusionModelCfg,
    seed: int,
) -> DiffusionTransformer:
    """Construct a diffusion model with initialization controlled by ``seed``."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return DiffusionTransformer(obs_dim=obs_dim, action_dim=action_dim, cfg=model_cfg)


class EmaModel:
    """Exponential moving average of the model weights (paper standard).

    ``decay`` is warmed up as ``min(decay, (1 + n) / (10 + n))`` so early
    updates track the fast-moving weights instead of the random init.
    """

    def __init__(self, model: DiffusionTransformer, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.n_updates = 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: DiffusionTransformer) -> None:
        self.n_updates += 1
        decay = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        for shadow, param in zip(self.module.parameters(), model.parameters(), strict=True):
            shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for shadow, buffer in zip(self.module.buffers(), model.buffers(), strict=True):
            shadow.copy_(buffer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.module.state_dict(),
            "decay": self.decay,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if float(payload["decay"]) != self.decay:
            raise ValueError("resume EMA decay conflicts with the frozen configuration")
        self.module.load_state_dict(payload["model_state"])
        self.n_updates = int(payload["n_updates"])


@dataclass(frozen=True)
class EpochStats:
    """Mean losses of one training epoch (validation only when it ran)."""

    epoch: int
    train_mse: float
    n_batches: int
    lr: float
    val_sampled_l1: float | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_mse": self.train_mse,
            "batch_count": self.n_batches,
            "learning_rate": self.lr,
            "sampled_validation_l1": self.val_sampled_l1,
            "duration_s": self.duration_s,
        }


@dataclass
class TrainHistory:
    """Per-epoch stats plus the best-validation bookkeeping."""

    epochs: list[EpochStats] = field(default_factory=list)
    best_epoch: int = -1
    best_val_l1: float = float("inf")
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [stats.to_dict() for stats in self.epochs],
            "best_epoch": self.best_epoch,
            "best_value": self.best_val_l1 if self.best_epoch >= 0 else None,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainHistory:
        epochs = [
            EpochStats(
                epoch=int(entry["epoch"]),
                train_mse=float(entry["train_mse"]),
                n_batches=int(entry.get("batch_count", entry.get("n_batches"))),
                lr=float(entry.get("learning_rate", entry.get("lr"))),
                val_sampled_l1=(
                    None
                    if entry.get("sampled_validation_l1", entry.get("val_sampled_l1")) is None
                    else float(entry.get("sampled_validation_l1", entry.get("val_sampled_l1")))
                ),
                duration_s=float(entry.get("duration_s", 0.0)),
            )
            for entry in payload.get("epochs", [])
        ]
        best_epoch = int(payload.get("best_epoch", -1))
        best_value = payload.get("best_value", payload.get("best_val_l1"))
        return cls(
            epochs=epochs,
            best_epoch=best_epoch,
            best_val_l1=float("inf") if best_value is None else float(best_value),
            duration_s=float(payload.get("duration_s", 0.0)),
        )


def train_diffusion(
    model: DiffusionTransformer,
    scheduler,
    make_train_batches: TrainBatchFactory,
    cfg: DiffusionTrainCfg,
    make_val_batches: ValBatchFactory | None = None,
    on_epoch: Callable[[EpochStats, bool], None] | None = None,
    ema: EmaModel | None = None,
    resume_state: dict[str, Any] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> TrainHistory:
    """Train ``model`` in place; returns the loss history.

    ``scheduler`` is the training DDPM schedule (``make_train_scheduler``).
    ``ema``, when given, is updated after every optimizer step and is the
    model the validation metric evaluates — the caller owns it and checkpoints
    ``ema.module``. ``on_epoch(stats, is_best)`` fires after every epoch;
    ``is_best`` is True when this epoch produced a new best sampled val L1
    (evaluated every ``cfg.val_every`` epochs and on the final epoch; without
    a validation factory the train MSE stands in).
    """
    if resume_state is None:
        torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    if ema is not None:
        ema.module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # One probe pass to size the cosine schedule (batch factories are cheap
    # numpy generators; the probe does not consume training randomness).
    steps_per_epoch = sum(1 for _ in make_train_batches(0))
    if steps_per_epoch == 0:
        raise ValueError("train batch factory yielded no batches at epoch 0")
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _lr_lambda(cfg, total_steps=cfg.epochs * steps_per_epoch),
    )

    history = (
        TrainHistory.from_dict(resume_state["history"])
        if resume_state is not None
        else TrainHistory()
    )
    start_epoch = int(resume_state["next_epoch"]) if resume_state is not None else 0
    global_step = int(resume_state["global_step"]) if resume_state is not None else 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        lr_scheduler.load_state_dict(resume_state["lr_scheduler_state"])
        if ema is None and resume_state.get("ema_state") is not None:
            raise ValueError("resume state contains EMA but the frozen config disables it")
        if ema is not None:
            if resume_state.get("ema_state") is None:
                raise ValueError("resume state is missing Diffusion EMA")
            ema.load_state_dict(resume_state["ema_state"])
        restore_rng_states(resume_state["rng_states"])
    elif on_checkpoint is not None:
        on_checkpoint(
            _training_state(
                optimizer,
                lr_scheduler,
                ema,
                history,
                next_epoch=0,
                global_step=0,
            )
        )

    for epoch in range(start_epoch, cfg.epochs):
        epoch_started = perf_counter()
        model.train()
        mse_sum = 0.0
        n_batches = 0
        for batch in make_train_batches(epoch):
            tensors = _to_device(collate_torch(batch), device)
            losses = diffusion_loss(
                model, scheduler, tensors["actions"], tensors["obs"], tensors["is_pad"]
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update(model)
            mse_sum += float(losses["mse"].detach())
            n_batches += 1
        if n_batches == 0:
            raise ValueError(f"train batch factory yielded no batches at epoch {epoch}")

        val_sampled_l1: float | None = None
        is_val_epoch = (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1
        if make_val_batches is not None and is_val_epoch:
            eval_model = ema.module if ema is not None else model
            val_sampled_l1 = evaluate_sampled_l1(
                eval_model,
                make_val_batches(),
                device,
                num_inference_steps=cfg.val_inference_steps,
            )

        stats = EpochStats(
            epoch=epoch,
            train_mse=mse_sum / n_batches,
            n_batches=n_batches,
            lr=float(optimizer.param_groups[0]["lr"]),
            val_sampled_l1=val_sampled_l1,
            duration_s=perf_counter() - epoch_started,
        )
        history.epochs.append(stats)
        history.duration_s += stats.duration_s

        selection = (
            val_sampled_l1
            if val_sampled_l1 is not None
            else (None if make_val_batches else stats.train_mse)
        )
        is_best = selection is not None and selection < history.best_val_l1
        if is_best:
            history.best_val_l1 = float(selection)
            history.best_epoch = epoch
        if on_epoch is not None:
            on_epoch(stats, is_best)
        if on_checkpoint is not None:
            on_checkpoint(
                _training_state(
                    optimizer,
                    lr_scheduler,
                    ema,
                    history,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                )
            )
    return history


def _training_state(
    optimizer,
    lr_scheduler,
    ema: EmaModel | None,
    history: TrainHistory,
    *,
    next_epoch: int,
    global_step: int,
) -> dict[str, Any]:
    return {
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "history": history.to_dict(),
        "rng_states": capture_rng_states(),
        "ema_state": ema.state_dict() if ema is not None else None,
    }


def evaluate_sampled_l1(
    model: DiffusionTransformer,
    batches: Iterable[dict],
    device: torch.device,
    num_inference_steps: int,
    seed: int = VAL_SAMPLING_SEED,
) -> float:
    """Masked L1 between DDIM-sampled chunks and targets (normalized units).

    A fixed torch.Generator seed makes the metric reproducible across epochs,
    so best-checkpoint selection is not dominated by sampling noise.
    """
    model.eval()
    inference = make_inference_scheduler(model.cfg, "ddim", num_inference_steps)
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            tensors = _to_device(collate_torch(batch), device)
            sampled = sample_actions(
                model,
                inference,
                tensors["obs"],
                model.cfg.horizon,
                model.action_dim,
                generator=generator,
            )
            valid = ~tensors["is_pad"].to(torch.bool)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            per_step = (sampled - tensors["actions"]).abs().mean(dim=-1)
            total += float((per_step * valid).sum())
            count += n_valid
    if count == 0:
        raise ValueError("validation batches contained no unpadded steps")
    return total / count


def _lr_lambda(cfg: DiffusionTrainCfg, total_steps: int):
    warmup = cfg.lr_warmup_steps

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if cfg.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return schedule


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


__all__ = [
    "VAL_SAMPLING_SEED",
    "EmaModel",
    "EpochStats",
    "TrainHistory",
    "evaluate_sampled_l1",
    "make_seeded_model",
    "train_diffusion",
]
