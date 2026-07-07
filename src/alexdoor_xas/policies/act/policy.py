"""Rollout-facing ACT policy wrapper: normalization + chunk-source factory.

Bridges the trained :class:`ACTModel` to the adapter-v1 rollout driver
(``adapters/rollout.rollout_chunks``) without importing it — the adapters
never import policies and vice versa; scripts compose the two. The env is
duck-typed through the frozen Phase 2 accessor surface (``proxy_pose_w`` /
``hinge_state`` / optional ``contact_sensed``), so the pure test fakes and
both Isaac envs work unchanged. No Isaac imports.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from alexdoor_xas.dataset import OBS_PRESETS, DatasetNormStats
from alexdoor_xas.policies.act.checkpoint import load_checkpoint
from alexdoor_xas.policies.act.model import ACTModel

OBS_CLIP = 10.0
"""Normalized-observation clip: near-constant training dims have their std
floored at 1e-8, so a small absolute rollout deviation would otherwise map to
an enormous normalized value far outside anything the model saw."""

ROLLOUT_OBS_PRESETS = ("core", "core_contact")
"""Presets with a closed-loop env reader. ``alex_full`` training remains
possible offline, but its joint-state/force layout has no verified live
reader yet, so rollout refuses it rather than risk a silent mismatch."""


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def build_env_obs(env, preset: str) -> np.ndarray:
    """Read the frozen observation preset live from the env, dataset-ordered."""
    if preset not in OBS_PRESETS:
        raise ValueError(f"unknown obs preset {preset!r} (known: {sorted(OBS_PRESETS)})")
    if preset not in ROLLOUT_OBS_PRESETS:
        raise ValueError(
            f"obs preset {preset!r} has no closed-loop env reader "
            f"(supported: {list(ROLLOUT_OBS_PRESETS)})"
        )
    ee_pos, ee_quat = env.proxy_pose_w()
    angle, velocity = env.hinge_state()
    parts = [
        np.asarray(
            ee_pos.detach().cpu().numpy() if isinstance(ee_pos, torch.Tensor) else ee_pos,
            dtype=np.float64,
        ).reshape(-1)[:3],
        np.asarray(
            ee_quat.detach().cpu().numpy() if isinstance(ee_quat, torch.Tensor) else ee_quat,
            dtype=np.float64,
        ).reshape(-1)[:4],
        np.array([_scalar(angle), _scalar(velocity)], dtype=np.float64),
    ]
    if preset == "core_contact":
        if not hasattr(env, "contact_sensed"):
            raise ValueError(
                "obs preset 'core_contact' needs env.contact_sensed(); "
                "this env does not expose force contact sensing"
            )
        parts.append(np.array([_scalar(env.contact_sensed())], dtype=np.float64))
    return np.concatenate(parts)


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
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> ActPolicy:
        loaded = load_checkpoint(path, map_location=device)
        policy = cls(loaded.model, loaded.stats, device=device)
        policy.checkpoint_config = loaded.config
        policy.checkpoint_meta = loaded.meta
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


def stop_on_hinge_angle(source: Callable, threshold_rad: float) -> Callable:
    """End the rollout once the door is open past ``threshold_rad``.

    The demos end when the scripted FSM completes, so a learned policy has no
    in-distribution behavior after task completion — left running, the
    extrapolating arm can knock the door shut again. This wrapper terminates
    at the first source query (chunk boundary) where the hinge angle has
    passed the threshold, bounding post-task extrapolation the same way the
    scripted episode termination does.
    """

    def wrapped(ctx):
        if ctx.hinge_angle_rad >= threshold_rad:
            return None
        return source(ctx)

    return wrapped


__all__ = [
    "OBS_CLIP",
    "ROLLOUT_OBS_PRESETS",
    "ActPolicy",
    "act_chunk_source",
    "build_env_obs",
    "stop_on_hinge_angle",
]
