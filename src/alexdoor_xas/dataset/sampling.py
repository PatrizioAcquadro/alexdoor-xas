"""Chunk sampling and seeded batching for learned baselines (Phase 3.0).

A2/A3/A1 samples follow the ACT convention: the observation at tick ``t`` plus
the action window ``t .. t+H-1``, zero-padded past the episode end with
``is_pad`` marking the padded slots. A4 episodes are already chunked
(symbolic per-phase structs, ~7 per episode — **not** a per-tick tensor
stream); they are exposed structurally plus an optional fixed-dim numeric
encoding per chunk (:func:`chunk_features`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import A4_PHASE_VOCAB, ObjectCentricChunk

from .loader import DEFAULT_OBS_PRESET, A4EpisodeRecord, EpisodeDataset, obs_matrix

A4_FEATURE_DIM = len(A4_PHASE_VOCAB) + 5
"""Per-chunk numeric encoding: phase one-hot + contact_target_panel (3) +
motion_hinge_delta_rad + duration_s."""


@dataclass(frozen=True)
class ChunkSample:
    """One training sample: obs at t + the H-step action window from t."""

    obs: np.ndarray  # (obs_dim,)
    actions: np.ndarray  # (H, D), zero-padded past the episode end
    is_pad: np.ndarray  # (H,) bool, True where the action is padding
    episode_id: str
    t_index: int
    t_s: float


class ChunkSampler:
    """Enumerate every (episode, tick) sample of a split at a fixed horizon."""

    def __init__(
        self,
        dataset: EpisodeDataset,
        horizon: int,
        obs_preset: str = DEFAULT_OBS_PRESET,
        episode_ids: list[str] | None = None,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.dataset = dataset
        self.horizon = horizon
        self.obs_preset = obs_preset
        ids = dataset.episode_ids if episode_ids is None else episode_ids
        self._records = [dataset.by_id(episode_id) for episode_id in ids]
        if not self._records:
            raise ValueError("sampler has no episodes")
        self._obs = [obs_matrix(record, obs_preset) for record in self._records]
        self._index: list[tuple[int, int]] = [
            (rec_idx, t)
            for rec_idx, record in enumerate(self._records)
            for t in range(record.n_steps)
        ]

    @property
    def obs_dim(self) -> int:
        return int(self._obs[0].shape[1])

    @property
    def action_dim(self) -> int:
        return self._records[0].action_dim

    def __len__(self) -> int:
        return len(self._index)

    def sample(self, index: int) -> ChunkSample:
        rec_idx, t = self._index[index]
        record = self._records[rec_idx]
        window = record.actions[t : t + self.horizon]
        n_valid = window.shape[0]
        actions = np.zeros((self.horizon, record.action_dim), dtype=np.float64)
        actions[:n_valid] = window
        is_pad = np.arange(self.horizon) >= n_valid
        return ChunkSample(
            obs=self._obs[rec_idx][t],
            actions=actions,
            is_pad=is_pad,
            episode_id=record.episode_id,
            t_index=t,
            t_s=float(record.t[t]),
        )


class BatchIterator:
    """Seeded single-pass batch iterator over a :class:`ChunkSampler`.

    Yields dict batches: ``obs (B, obs_dim)``, ``actions (B, H, D)``,
    ``is_pad (B, H)``, ``t (B,)``, ``episode_ids`` (list of str), and the
    dataset's ``action_space`` tag. Iterating twice with the same seed yields
    identical batches.
    """

    def __init__(
        self,
        sampler: ChunkSampler,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        order = np.random.default_rng(self.seed).permutation(len(self.sampler))
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if self.drop_last and len(chosen) < self.batch_size:
                return
            samples = [self.sampler.sample(int(i)) for i in chosen]
            yield {
                "obs": np.stack([s.obs for s in samples]),
                "actions": np.stack([s.actions for s in samples]),
                "is_pad": np.stack([s.is_pad for s in samples]),
                "t_index": np.array([s.t_index for s in samples], dtype=np.int64),
                "t": np.array([s.t_index for s in samples], dtype=np.int64),
                "t_s": np.array([s.t_s for s in samples], dtype=np.float64),
                "episode_ids": [s.episode_id for s in samples],
                "action_space": self.sampler.dataset.action_space,
            }

    def __len__(self) -> int:
        n_batches, remainder = divmod(len(self.sampler), self.batch_size)
        return n_batches if (self.drop_last or remainder == 0) else n_batches + 1


def chunk_features(chunk: ObjectCentricChunk, control_dt: float) -> np.ndarray:
    """Fixed-dim numeric encoding of one A4 chunk (:data:`A4_FEATURE_DIM`,)."""
    if chunk.phase not in A4_PHASE_VOCAB:
        raise ValueError(f"unknown A4 phase {chunk.phase!r} (vocabulary: {A4_PHASE_VOCAB})")
    one_hot = np.zeros(len(A4_PHASE_VOCAB), dtype=np.float64)
    one_hot[A4_PHASE_VOCAB.index(chunk.phase)] = 1.0
    return np.concatenate(
        [
            one_hot,
            np.asarray(chunk.contact_target_panel, dtype=np.float64),
            [chunk.motion_hinge_delta_rad, chunk.duration_ticks * control_dt],
        ]
    )


def episode_chunk_features(record: A4EpisodeRecord) -> np.ndarray:
    """Encode one A4 episode's chunk log as a ``(C, A4_FEATURE_DIM)`` matrix."""
    if not record.chunks:
        return np.zeros((0, A4_FEATURE_DIM), dtype=np.float64)
    return np.stack([chunk_features(chunk, record.control_dt) for chunk in record.chunks])


def collate_torch(batch: dict[str, Any]):  # pragma: no cover - exercised by the gate
    """Convert a numpy batch to float32 torch tensors (torch is optional)."""
    import torch

    return {
        key: torch.as_tensor(value, dtype=torch.float32)
        if isinstance(value, np.ndarray) and value.dtype != np.int64
        else (torch.as_tensor(value) if isinstance(value, np.ndarray) else value)
        for key, value in batch.items()
    }


__all__ = [
    "A4_FEATURE_DIM",
    "A4_PHASE_VOCAB",
    "BatchIterator",
    "ChunkSample",
    "ChunkSampler",
    "chunk_features",
    "collate_torch",
    "episode_chunk_features",
]
