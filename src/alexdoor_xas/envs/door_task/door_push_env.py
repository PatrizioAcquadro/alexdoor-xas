"""Phase 2 door-push DirectRLEnv: door fixture + velocity-driven proxy end-effector.

The action is a 6-dim end-effector delta ``(dx, dy, dz, drx, dry, drz)`` per
control tick (the A2 representation). It is executed by commanding the dynamic,
gravity-free proxy sphere with the equivalent root velocity on every physics
sub-step, so contact with the door is resolved by the solver rather than by
kinematic teleports. The sphere stands in for an Alex hand until the Alex
adapter exists (see
:data:`~alexdoor_xas.envs.door_task.door_push_env_cfg.PROXY_EE_ROBOT_TAG`).

Known backend limitation: articulation link/root pose reads (``body_pos_w``,
``root_pos_w``) return zeros in this Isaac Lab release's direct-env context, so
the door frame is read from the USD stage at reset instead. That is valid here
because the door frame is fixed to the world (drift-checked by
``scripts/verify_door_task_scene.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from alexdoor_xas.assets.door_task import ensure_door_task_usd

from .door_env import resolve_hinge_joint_id
from .door_push_env_cfg import OBSERVATION_TERMS

if TYPE_CHECKING:
    from .door_push_env_cfg import DoorPushEnvCfg

DOORFRAME_PRIM_TEMPLATE = "/World/envs/env_{index}/DoorTaskScene/DoorTaskDoor/Doorframe"


def read_doorframe_from_stage(env_id: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """World pose of one env's ``Doorframe`` prim, read from the USD stage.

    Returns ``(pos, quat_xyzw)``. Used instead of live articulation pose reads,
    which return zeros in this Isaac Lab build's direct-env context (see module
    docstring). Valid only after ``AppLauncher``.
    """
    import omni.usd  # noqa: PLC0415 - Kit runtime import, valid after AppLauncher.
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_path = DOORFRAME_PRIM_TEMPLATE.format(index=env_id)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"door frame prim not found on stage: {prim_path}")
    transform = cache.GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    quat = transform.RemoveScaleShear().ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        (translation[0], translation[1], translation[2]),
        (imaginary[0], imaginary[1], imaginary[2], quat.GetReal()),
    )


class DoorPushEnv(DirectRLEnv):
    """Single-door push task driven by an externally scripted (or learned) policy."""

    cfg: DoorPushEnvCfg

    def __init__(self, cfg: DoorPushEnvCfg, render_mode: str | None = None, **kwargs):
        usd_path = ensure_door_task_usd(
            door_yaw_rad=cfg.door_yaw_rad, door_xy_offset_m=tuple(cfg.door_offset_xy)
        )
        cfg.door_task_scene.spawn.usd_path = str(usd_path)

        super().__init__(cfg, render_mode, **kwargs)

        self._hinge_joint_id = resolve_hinge_joint_id(
            list(self._door.joint_names), self.cfg.hinge_joint_name
        )
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._commanded_root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self._zero_reward = torch.zeros(self.num_envs, device=self.device)
        self._never_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._door_frame_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._door_frame_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._read_door_frame_from_stage(self._all_env_ids)

    # -- scene ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        if self.cfg.door_task_scene.spawn is None:
            raise RuntimeError("DoorPushEnvCfg.door_task_scene.spawn must be configured")

        self.cfg.door_task_scene.spawn.func(
            self.cfg.door_task_scene.prim_path,
            self.cfg.door_task_scene.spawn,
            translation=self.cfg.door_task_scene.init_state.pos,
            orientation=self.cfg.door_task_scene.init_state.rot,
        )
        self._door = Articulation(self.cfg.door)
        self._proxy = RigidObject(self.cfg.proxy_ee)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["door"] = self._door
        self.scene.rigid_objects["proxy_ee"] = self._proxy

    # -- control ----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.cfg.action_space)
        if actions.shape != expected_shape:
            raise ValueError(
                f"DoorPushEnv expects actions with shape {expected_shape}, "
                f"got {tuple(actions.shape)}"
            )
        clamped = actions.clone()
        clamped[:, :3] = clamped[:, :3].clamp(-self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
        clamped[:, 3:] = clamped[:, 3:].clamp(
            -self.cfg.max_rot_delta_rad, self.cfg.max_rot_delta_rad
        )
        self.actions = clamped
        # A delta per control tick corresponds to a constant root velocity held
        # for the whole tick (all physics sub-steps).
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        self._commanded_root_vel = clamped / control_dt

    def _apply_action(self) -> None:
        self._proxy.write_root_velocity_to_sim_index(
            root_velocity=self._commanded_root_vel, env_ids=self._all_env_ids
        )

    # -- MDP terms ---------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        hinge_pos = self._door.data.joint_pos.torch[:, self._hinge_joint_id].unsqueeze(-1)
        hinge_vel = self._door.data.joint_vel.torch[:, self._hinge_joint_id].unsqueeze(-1)
        ee_pos = self._proxy.data.root_pos_w.torch - self.scene.env_origins
        ee_quat = self._proxy.data.root_quat_w.torch
        obs = torch.cat((hinge_pos, hinge_vel, ee_pos, ee_quat), dim=-1)
        if not torch.isfinite(obs).all():
            raise RuntimeError(
                f"door push observations {OBSERVATION_TERMS} contain non-finite values: "
                f"{obs.detach().cpu()}"
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
            env_ids = self._all_env_ids
        env_ids = torch.as_tensor(env_ids, device=self.device)

        super()._reset_idx(env_ids)

        # Only the hinge state is reset. The door root is world-anchored by the
        # task layer's FixDoorframe joint; writing cfg-default root poses here
        # would teleport the articulation to the env origin and fight that
        # anchor (the articulation cfg intentionally has no init_state).
        joint_pos = self._door.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(self._door.data.default_joint_vel.torch[env_ids])
        self._door.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._door.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        proxy_pose = self._proxy.data.default_root_pose.torch[env_ids].clone()
        proxy_pose[:, :3] += self.scene.env_origins[env_ids]
        self._write_proxy_pose(proxy_pose, env_ids)
        self._commanded_root_vel[env_ids] = 0.0

    # -- Phase 2 state accessors (used by the scripted controller / recorder) ----

    def door_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the ``Doorframe`` body: the hinge axis passes through it.

        Read from the USD stage at reset (world-fixed frame; see module docstring).
        """
        return self._door_frame_pos_w.clone(), self._door_frame_quat_w.clone()

    def hinge_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Hinge angle (rad) and angular velocity (rad/s), shape ``(num_envs,)``."""
        return (
            self._door.data.joint_pos.torch[:, self._hinge_joint_id].clone(),
            self._door.data.joint_vel.torch[:, self._hinge_joint_id].clone(),
        )

    def proxy_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and ``(x, y, z, w)`` orientation of the proxy EE."""
        return (
            self._proxy.data.root_pos_w.torch.clone(),
            self._proxy.data.root_quat_w.torch.clone(),
        )

    def set_proxy_pose(
        self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Teleport the proxy EE (data engine: fixed or seeded-jittered starts)."""
        if env_ids is None:
            env_ids = self._all_env_ids
        root_pose = torch.cat(
            (
                pos_w.to(self.device).reshape(len(env_ids), 3),
                quat_w.to(self.device).reshape(len(env_ids), 4),
            ),
            dim=-1,
        )
        self._write_proxy_pose(root_pose, env_ids)

    def _write_proxy_pose(self, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        root_vel = torch.zeros((len(env_ids), 6), device=self.device)
        self._proxy.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._proxy.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)

    def _read_door_frame_from_stage(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            translation, quat_xyzw = read_doorframe_from_stage(env_id)
            self._door_frame_pos_w[env_id] = torch.tensor(
                translation, dtype=torch.float32, device=self.device
            )
            self._door_frame_quat_w[env_id] = torch.tensor(
                quat_xyzw, dtype=torch.float32, device=self.device
            )


__all__ = ["DOORFRAME_PRIM_TEMPLATE", "DoorPushEnv", "read_doorframe_from_stage"]
