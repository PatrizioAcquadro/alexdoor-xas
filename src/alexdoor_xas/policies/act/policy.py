"""Rollout-facing ACT policy wrapper: normalization + chunk-source factory.

Bridges the trained :class:`ACTModel` to the adapter-v1 rollout driver
(``adapters/rollout.rollout_chunks``) without importing it — the adapters
never import policies and vice versa; scripts compose the two. The live obs
readers and the success-stop wrapper are shared across chunk policies and
live in ``policies.common.obs`` (re-exported here for compatibility).
No Isaac imports.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import (
    RobotAssetRef,
    assert_checkpoint_runtime_compatible,
)
from alexdoor_xas.dataset import DatasetNormStats
from alexdoor_xas.policies.act.checkpoint import load_checkpoint
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.common.obs import (
    OBS_CLIP,
    ROLLOUT_OBS_PRESETS,
    build_env_obs,
    stop_on_hinge_angle,
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
            raise ValueError(
                f"norm stats obs dim {stats.obs.dim} != model obs dim {model.obs_dim}"
            )
        if stats.action.dim != model.action_dim:
            raise ValueError(
                f"norm stats action dim {stats.action.dim} != model action dim "
                f"{model.action_dim}"
            )
        self.model = model
        self.stats = stats
        self.device = torch.device(device)
        self.obs_clip = obs_clip
        self.checkpoint_config: dict | None = None
        self.checkpoint_meta: dict | None = None
        self.checkpoint_split_episode_ids: dict[str, tuple[str, ...]] = {}
        self.checkpoint_provenance: dict = {}
        self.robot_asset: RobotAssetRef | None = None
        self.robot_compatibility_label: str | None = None
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        *,
        runtime_asset: RobotAssetRef | None = None,
        allow_cross_model_evaluation: bool = False,
    ) -> ActPolicy:
        loaded = load_checkpoint(path, map_location=device)
        policy = cls(loaded.model, loaded.stats, device=device)
        policy.checkpoint_config = loaded.config
        policy.checkpoint_meta = loaded.meta
        policy.checkpoint_split_episode_ids = loaded.split_episode_ids
        policy.checkpoint_provenance = loaded.provenance
        policy.robot_asset = loaded.robot_asset
        dataset = loaded.config.get("dataset", {})
        checkpoint_is_v2 = (
            isinstance(dataset, dict) and dataset.get("task") == paths.ALEX_V2_TASK
        )
        if checkpoint_is_v2 and runtime_asset is None:
            raise ValueError("Alex V2 policy loading requires a runtime robot asset")
        if runtime_asset is not None:
            policy.robot_compatibility_label = assert_checkpoint_runtime_compatible(
                loaded.robot_asset,
                runtime_asset,
                allow_cross_model_evaluation=allow_cross_model_evaluation,
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
            raise ValueError(
                f"expected obs of dim {self.model.obs_dim}, got {obs.shape[0]}"
            )
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
    if preset not in ROLLOUT_OBS_PRESETS:
        raise ValueError(
            f"obs preset {preset!r} has no closed-loop env reader "
            f"(supported: {list(ROLLOUT_OBS_PRESETS)})"
        )

    if not temporal_ensemble:

        def source(ctx):
            del ctx
            return policy.predict(build_env_obs(env, preset))

        return source

    pending: list[np.ndarray] = []  # oldest first; each holds its remaining future rows

    def ensemble_source(ctx):
        del ctx
        pending.append(policy.predict(build_env_obs(env, preset)))
        current = np.stack([chunk[0] for chunk in pending])
        weights = np.exp(-ensemble_m * np.arange(len(pending), dtype=np.float64))
        action = (current * weights[:, None]).sum(axis=0) / weights.sum()
        # Consume this tick's row from every buffered chunk; drop exhausted ones.
        pending[:] = [chunk[1:] for chunk in pending if chunk.shape[0] > 1]
        return action.reshape(1, -1)

    return ensemble_source


__all__ = [
    "OBS_CLIP",
    "ROLLOUT_OBS_PRESETS",
    "ActPolicy",
    "act_chunk_source",
    "build_env_obs",
    "stop_on_hinge_angle",
]
