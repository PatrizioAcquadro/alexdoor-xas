"""Rollout-facing ACT policy normalization and chunk-source factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from alexdoor_xas.assets.alex_v2_contract import (
    RobotAssetRef,
    assert_checkpoint_runtime_compatible,
)
from alexdoor_xas.dataset.normalize import DatasetNormStats
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.common.checkpoint import (
    ACT_CHECKPOINT_FORMAT,
    load_checkpoint_payload,
)
from alexdoor_xas.policies.common.obs import (
    OBS_CLIP,
    build_rollout_obs,
    read_door_pose_obs,
    validate_obs_preset,
)


class ActPolicy:
    """A trained ACT model plus the normalization stats it was trained with."""

    def __init__(
        self,
        model: ACTModel,
        stats: DatasetNormStats,
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
        self.device = torch.device(device)
        self.obs_clip = obs_clip
        self.robot_compatibility_label: str | None = None
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        *,
        runtime_asset: RobotAssetRef,
    ) -> ActPolicy:
        loaded = load_checkpoint_payload(path, ACT_CHECKPOINT_FORMAT, "ACT", device)
        try:
            model_cfg = ActModelCfg(**loaded.model_cfg)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid ACT checkpoint {path}: {error}") from error
        model = ACTModel(loaded.obs_dim, loaded.action_dim, model_cfg)
        model.load_state_dict(loaded.state_dict)
        policy = cls(model, loaded.stats, device=device)
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
        return self.model.cfg.chunk_size

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """One denormalized action chunk ``(H, D)`` for one raw observation."""
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape[0] != self.model.obs_dim:
            raise ValueError(f"expected obs of dim {self.model.obs_dim}, got {obs.shape[0]}")
        normalized = np.clip(self.stats.obs.normalize(obs), -self.obs_clip, self.obs_clip)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        a_hat = self.model.predict(tensor.reshape(1, -1))[0].cpu().numpy()
        return self.stats.action.denormalize(a_hat)


def act_chunk_source(
    policy: ActPolicy,
    env,
    obs_preset: str | None = None,
    temporal_ensemble: bool = False,
    ensemble_m: float = 0.01,
) -> Callable:
    """Adapt ``policy`` to the ``rollout_chunks`` chunk-source protocol.

    Default mode reads a fresh observation and emits the full ``(H, 6)`` chunk
    (the driver executes it delta-by-delta, so the policy is re-queried every
    ``H`` ticks). Temporal-ensemble mode emits a single ``(1, 6)`` action per
    call — the exponentially weighted average (``w_i = exp(-m * i)``, ``i = 0``
    oldest) of every past chunk's prediction for the current tick, so the
    policy is queried every tick, per the ACT paper.
    """
    preset = obs_preset or policy.obs_preset
    validate_obs_preset(preset)
    door_pose = read_door_pose_obs(env) if preset == "core_door_pose" else None

    if not temporal_ensemble:

        def source(ctx):
            return policy.predict(build_rollout_obs(ctx, preset, door_pose))

        return source

    pending: list[np.ndarray] = []  # oldest first; each holds its remaining future rows

    def ensemble_source(ctx):
        pending.append(policy.predict(build_rollout_obs(ctx, preset, door_pose)))
        current = np.stack([chunk[0] for chunk in pending])
        weights = np.exp(-ensemble_m * np.arange(len(pending), dtype=np.float64))
        action = (current * weights[:, None]).sum(axis=0) / weights.sum()
        # Consume this tick's row from every buffered chunk; drop exhausted ones.
        pending[:] = [chunk[1:] for chunk in pending if chunk.shape[0] > 1]
        return action.reshape(1, -1)

    return ensemble_source
