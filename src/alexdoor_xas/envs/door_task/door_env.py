"""Minimal Isaac Lab DirectRLEnv shell for the single-door task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from alexdoor_xas.assets.door_task import ensure_door_task_usd

from .door_env_cfg import OBSERVATION_TERMS
from .door_runtime import resolve_hinge_joint_id

if TYPE_CHECKING:
    from .door_env_cfg import DoorTaskEnvCfg


class DoorTaskEnv(DirectRLEnv):
    """A door-only DirectRLEnv used to gate Isaac Lab task instantiation."""

    cfg: DoorTaskEnvCfg

    def __init__(self, cfg: DoorTaskEnvCfg, render_mode: str | None = None, **kwargs):
        usd_path = ensure_door_task_usd()
        cfg.door_task_scene.spawn.usd_path = str(usd_path)

        super().__init__(cfg, render_mode, **kwargs)

        self._hinge_joint_id = self._resolve_hinge_joint_id()
        self.actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._zero_reward = torch.zeros(self.num_envs, device=self.device)
        self._never_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        if self.cfg.door_task_scene.spawn is None:
            raise RuntimeError("DoorTaskEnvCfg.door_task_scene.spawn must be configured")

        self.cfg.door_task_scene.spawn.func(
            self.cfg.door_task_scene.prim_path,
            self.cfg.door_task_scene.spawn,
            translation=self.cfg.door_task_scene.init_state.pos,
            orientation=self.cfg.door_task_scene.init_state.rot,
        )
        self._door = Articulation(self.cfg.door)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["door"] = self._door

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 1):
            expected_shape = (self.num_envs, 1)
            raise ValueError(
                f"DoorTaskEnv expects actions with shape {expected_shape}, "
                f"got {tuple(actions.shape)}"
            )
        self.actions = actions.clone().clamp(-self.cfg.action_clip, self.cfg.action_clip)

    def _apply_action(self) -> None:
        # The pre-Phase-2 shell deliberately stores but does not command actions.
        return

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._door.data.joint_pos.torch[:, self._hinge_joint_id]
        joint_vel = self._door.data.joint_vel.torch[:, self._hinge_joint_id]
        obs = torch.stack((joint_pos, joint_vel), dim=-1)
        if not torch.isfinite(obs).all():
            bad_obs = obs.detach().cpu()
            raise RuntimeError(
                f"door observations {OBSERVATION_TERMS} contain non-finite values: {bad_obs}"
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return self._zero_reward.clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._never_done.clone()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._door._ALL_INDICES

        super()._reset_idx(env_ids)

        # Only the hinge state is reset. The door root is world-anchored by the
        # task layer's FixDoorframe joint; writing cfg-default root poses here
        # would teleport the articulation to the env origin and fight that
        # anchor (the articulation cfg intentionally has no init_state).
        joint_pos = self._door.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(self._door.data.default_joint_vel.torch[env_ids])
        self._door.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._door.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _resolve_hinge_joint_id(self) -> int:
        return resolve_hinge_joint_id(list(self._door.joint_names), self.cfg.hinge_joint_name)

__all__ = ["DoorTaskEnv", "resolve_hinge_joint_id"]
