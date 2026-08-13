"""Supervised ACT training loop (torch only; no Isaac imports, no file I/O).

The loop consumes *batch factories* — callables producing an iterable of
numpy batch dicts in the Phase 3.0 ``BatchIterator`` layout (``obs``,
``actions``, ``is_pad``), already normalized — so it is unit-testable with
hand-built batches and the script layer owns dataset/paths/tracking concerns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from alexdoor_xas.dataset.sampling import collate_torch
from alexdoor_xas.policies.act.config import ActModelCfg, ActTrainCfg
from alexdoor_xas.policies.act.model import ACTModel, act_loss
from alexdoor_xas.policies.common.runs import capture_rng_states, restore_rng_states

TrainBatchFactory = Callable[[int], Iterable[dict]]
"""``epoch -> iterable of normalized numpy batch dicts`` (fresh shuffle per epoch)."""
ValBatchFactory = Callable[[], Iterable[dict]]


def make_seeded_model(
    obs_dim: int,
    action_dim: int,
    model_cfg: ActModelCfg,
    seed: int,
) -> ACTModel:
    """Construct an ACT model with initialization controlled by ``seed``."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return ACTModel(obs_dim=obs_dim, action_dim=action_dim, cfg=model_cfg)


@dataclass(frozen=True)
class EpochStats:
    """Mean losses of one training epoch (validation only when it ran)."""

    epoch: int
    train_l1: float
    train_kl: float
    train_loss: float
    n_batches: int
    val_l1: float | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_l1": self.train_l1,
            "kl": self.train_kl,
            "total_loss": self.train_loss,
            "batch_count": self.n_batches,
            "validation_l1": self.val_l1,
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
                train_l1=float(entry["train_l1"]),
                train_kl=float(entry.get("kl", entry.get("train_kl"))),
                train_loss=float(entry.get("total_loss", entry.get("train_loss"))),
                n_batches=int(entry.get("batch_count", entry.get("n_batches"))),
                val_l1=(
                    None
                    if entry.get("validation_l1", entry.get("val_l1")) is None
                    else float(entry.get("validation_l1", entry.get("val_l1")))
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


def train_act(
    model: ACTModel,
    make_train_batches: TrainBatchFactory,
    cfg: ActTrainCfg,
    make_val_batches: ValBatchFactory | None = None,
    on_epoch: Callable[[EpochStats, bool], None] | None = None,
    resume_state: dict[str, Any] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> TrainHistory:
    """Train ``model`` in place; returns the loss history.

    ``on_epoch(stats, is_best)`` fires after every epoch; ``is_best`` is True
    when this epoch produced a new best validation L1 (evaluated every
    ``cfg.val_every`` epochs and on the final epoch; without a validation
    factory the train loss stands in).
    """
    if resume_state is None:
        torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = (
        TrainHistory.from_dict(resume_state["history"])
        if resume_state is not None
        else TrainHistory()
    )
    start_epoch = int(resume_state["next_epoch"]) if resume_state is not None else 0
    global_step = int(resume_state["global_step"]) if resume_state is not None else 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        restore_rng_states(resume_state["rng_states"])
    elif on_checkpoint is not None:
        on_checkpoint(_training_state(optimizer, history, next_epoch=0, global_step=0))

    for epoch in range(start_epoch, cfg.epochs):
        epoch_started = perf_counter()
        model.train()
        sums = {"l1": 0.0, "kl": 0.0, "loss": 0.0}
        n_batches = 0
        for batch in make_train_batches(epoch):
            tensors = _to_device(collate_torch(batch), device)
            a_hat, mu, logvar = model(tensors["obs"], tensors["actions"], tensors["is_pad"])
            losses = act_loss(
                a_hat, tensors["actions"], tensors["is_pad"], mu, logvar, cfg.kl_weight
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1
            for key in sums:
                sums[key] += float(losses[key].detach())
            n_batches += 1
        if n_batches == 0:
            raise ValueError(f"train batch factory yielded no batches at epoch {epoch}")

        val_l1: float | None = None
        is_val_epoch = (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1
        if make_val_batches is not None and is_val_epoch:
            val_l1 = evaluate_l1(model, make_val_batches(), device)

        stats = EpochStats(
            epoch=epoch,
            train_l1=sums["l1"] / n_batches,
            train_kl=sums["kl"] / n_batches,
            train_loss=sums["loss"] / n_batches,
            n_batches=n_batches,
            val_l1=val_l1,
            duration_s=perf_counter() - epoch_started,
        )
        history.epochs.append(stats)
        history.duration_s += stats.duration_s

        selection = val_l1 if val_l1 is not None else (None if make_val_batches else stats.train_l1)
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
                    history,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                )
            )
    return history


def _training_state(
    optimizer, history: TrainHistory, *, next_epoch: int, global_step: int
) -> dict[str, Any]:
    return {
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": None,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "history": history.to_dict(),
        "rng_states": capture_rng_states(),
    }


def evaluate_l1(model: ACTModel, batches: Iterable[dict], device: torch.device) -> float:
    """Masked L1 of the z=0 policy prediction (matches inference conditions)."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            tensors = _to_device(collate_torch(batch), device)
            a_hat = model.predict(tensors["obs"])
            valid = ~tensors["is_pad"].to(torch.bool)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            per_step = (a_hat - tensors["actions"]).abs().mean(dim=-1)
            total += float((per_step * valid).sum())
            count += n_valid
    if count == 0:
        raise ValueError("validation batches contained no unpadded steps")
    return total / count


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


__all__ = ["EpochStats", "TrainHistory", "evaluate_l1", "make_seeded_model", "train_act"]
