"""Diffusion-specific normalization on top of the shared dataset plumbing.

Torch-free. The one behavioral difference from ACT is the action
normalization: DDPM sampling clips to [-1, 1] every step
(``clip_sample=True``), so actions are min-max scaled into [-1, 1] (Diffusion
Policy paper, appendix A.1) instead of z-scored — z-scoring would leave parts
of the action range unreachable after clipping. Observations stay z-scored
like ACT. Constant action dims (the recorded-but-never-actuated A2/A3
rotation deltas) use the paper's guard: shift to zero without scaling, so
they normalize and denormalize to exactly 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.policies.common.data import (
    PolicyData,
    load_policy_data,
)
from alexdoor_xas.policies.common.data import (
    make_eval_factory as _make_eval_factory,
)
from alexdoor_xas.policies.common.data import (
    make_train_factory as _make_train_factory,
)
from alexdoor_xas.policies.diffusion.config import DiffusionDatasetCfg

RANGE_EPS = 1e-8
"""A dimension whose train-split range is below this is treated as constant."""


@dataclass(frozen=True)
class MinMaxNormalizer:
    """Per-dimension min-max scaling to [-1, 1] with a constant-dim guard."""

    center: np.ndarray  # (D,)
    scale: np.ndarray  # (D,); 1.0 on constant dims (shift-to-zero, no scaling)

    @classmethod
    def from_norm_stats(cls, stats: NormStats, range_eps: float = RANGE_EPS) -> MinMaxNormalizer:
        low = np.asarray(stats.min, dtype=np.float64)
        high = np.asarray(stats.max, dtype=np.float64)
        span = high - low
        constant = span < range_eps
        center = np.where(constant, low, (low + high) / 2.0)
        scale = np.where(constant, 1.0, 2.0 / np.where(constant, 1.0, span))
        return cls(center=center, scale=scale)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.center) * self.scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) / self.scale + self.center


def load_diffusion_data(cfg: DiffusionDatasetCfg, **kwargs) -> PolicyData:
    """The shared dataset/splits/stats loader (staleness checks included)."""
    return load_policy_data(cfg, **kwargs)


def make_diffusion_normalizer(stats: DatasetNormStats):
    """Batch normalizer: z-score obs (ACT-identical), min-max actions."""
    action_minmax = MinMaxNormalizer.from_norm_stats(stats.action)

    def normalize(batch: dict[str, Any], batch_stats: DatasetNormStats) -> dict[str, Any]:
        normalized = dict(batch)
        normalized["obs"] = batch_stats.obs.normalize(batch["obs"])
        normalized["actions"] = action_minmax.normalize(batch["actions"])
        return normalized

    return normalize


def make_train_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...] | None = None,
):
    """Per-epoch reshuffled batches with diffusion normalization."""
    return _make_train_factory(
        data,
        chunk_size,
        batch_size,
        seed,
        episode_ids=episode_ids,
        normalize=make_diffusion_normalizer(data.stats),
    )


def make_eval_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...],
):
    """Fixed-order batches with diffusion normalization."""
    return _make_eval_factory(
        data,
        chunk_size,
        batch_size,
        seed,
        episode_ids=episode_ids,
        normalize=make_diffusion_normalizer(data.stats),
    )


__all__ = [
    "RANGE_EPS",
    "MinMaxNormalizer",
    "load_diffusion_data",
    "make_diffusion_normalizer",
    "make_eval_factory",
    "make_train_factory",
]
