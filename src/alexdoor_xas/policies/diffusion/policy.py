"""Rollout-facing Diffusion Policy normalization and chunk-source factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from alexdoor_xas.assets.alex_v2_contract import (
    RobotAssetRef,
    assert_checkpoint_runtime_compatible,
)
from alexdoor_xas.dataset import DatasetNormStats
from alexdoor_xas.policies.common.obs import (
    OBS_CLIP,
    ROLLOUT_OBS_PRESETS,
    build_rollout_obs,
    read_door_pose_obs,
)
from alexdoor_xas.policies.diffusion.checkpoint import load_checkpoint
from alexdoor_xas.policies.diffusion.data import MinMaxNormalizer
from alexdoor_xas.policies.diffusion.model import DiffusionTransformer
from alexdoor_xas.policies.diffusion.schedulers import make_inference_scheduler, sample_actions


class DiffusionPolicy:
    """A trained diffusion model plus its stats, sampler, and noise generator.

    Sampling is stochastic; :meth:`seed` resets the internal generator so a
    fixed-seed rollout is a meaningful determinism probe (DDIM runs with
    eta=0, DDPM draws its per-step noise from the same generator).
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        stats: DatasetNormStats,
        *,
        sampler: str = "ddpm",
        num_inference_steps: int = 100,
        device: str = "cpu",
        obs_clip: float = OBS_CLIP,
    ) -> None:
        if stats.obs.dim != model.obs_dim:
            raise ValueError(f"norm stats obs dim {stats.obs.dim} != model obs dim {model.obs_dim}")
        if stats.action.dim != model.action_dim:
            raise ValueError(
                f"norm stats action dim {stats.action.dim} != model action dim {model.action_dim}"
            )
        self.model = model
        self.stats = stats
        self.sampler = sampler
        self.num_inference_steps = num_inference_steps
        self.device = torch.device(device)
        self.obs_clip = obs_clip
        self.checkpoint_config: dict | None = None
        self.checkpoint_meta: dict | None = None
        self.checkpoint_format: str | None = None
        self.robot_asset: RobotAssetRef | None = None
        self.robot_compatibility_label: str | None = None
        self._action_minmax = MinMaxNormalizer.from_norm_stats(stats.action)
        self._scheduler = make_inference_scheduler(model.cfg, sampler, num_inference_steps)
        self._generator = torch.Generator()
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        sampler: str = "ddpm",
        num_inference_steps: int = 100,
        *,
        runtime_asset: RobotAssetRef,
    ) -> DiffusionPolicy:
        loaded = load_checkpoint(path, map_location=device)
        policy = cls(
            loaded.model,
            loaded.stats,
            sampler=sampler,
            num_inference_steps=num_inference_steps,
            device=device,
        )
        policy.checkpoint_config = loaded.config
        policy.checkpoint_meta = loaded.meta
        policy.checkpoint_format = loaded.checkpoint_format
        policy.robot_asset = loaded.robot_asset
        policy.robot_compatibility_label = assert_checkpoint_runtime_compatible(
            loaded.robot_asset,
            runtime_asset,
        )
        return policy

    @property
    def action_space(self) -> str:
        return self.stats.action_space

    @property
    def obs_preset(self) -> str:
        return self.stats.obs_preset

    @property
    def chunk_size(self) -> int:
        return self.model.cfg.horizon

    def seed(self, seed: int) -> None:
        """Reset the sampling generator (call once per rollout for determinism)."""
        self._generator = torch.Generator().manual_seed(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """One denormalized action chunk ``(H, D)`` for one raw observation."""
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape[0] != self.model.obs_dim:
            raise ValueError(f"expected obs of dim {self.model.obs_dim}, got {obs.shape[0]}")
        normalized = np.clip(self.stats.obs.normalize(obs), -self.obs_clip, self.obs_clip)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        sampled = (
            sample_actions(
                self.model,
                self._scheduler,
                tensor.reshape(1, -1),
                self.model.cfg.horizon,
                self.model.action_dim,
                generator=self._generator,
            )[0]
            .cpu()
            .numpy()
        )
        return self._action_minmax.denormalize(sampled)


def diffusion_chunk_source(
    policy: DiffusionPolicy,
    env,
    obs_preset: str | None = None,
    n_action_steps: int | None = None,
) -> Callable:
    """Adapt ``policy`` to the ``rollout_chunks`` chunk-source protocol.

    Receding-horizon execution per the Diffusion Policy paper: each call reads
    a fresh observation, samples a full ``(Tp, 6)`` chunk, and emits only its
    first ``n_action_steps`` rows (Ta), so the driver re-queries the policy
    every Ta ticks. ``n_action_steps=None`` executes the whole chunk.
    """
    preset = obs_preset or policy.obs_preset
    if preset not in ROLLOUT_OBS_PRESETS:
        raise ValueError(
            f"obs preset {preset!r} has no closed-loop env reader "
            f"(supported: {list(ROLLOUT_OBS_PRESETS)})"
        )
    steps = policy.chunk_size if n_action_steps is None else int(n_action_steps)
    if not 1 <= steps <= policy.chunk_size:
        raise ValueError(f"n_action_steps must be in [1, {policy.chunk_size}], got {steps}")
    door_pose = read_door_pose_obs(env) if preset == "core_door_pose" else None

    def source(ctx):
        return policy.predict(build_rollout_obs(ctx, preset, door_pose))[:steps]

    return source


__all__ = ["DiffusionPolicy", "diffusion_chunk_source"]
