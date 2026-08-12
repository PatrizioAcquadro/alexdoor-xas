"""Robot-agnostic door-push loop for a fixed-base articulated IK executor.

The action contract is a 6-dim EE delta ``(dx, dy, dz, drx, dry, drz)`` per
control tick (A2). The clamped translation becomes a relative position command
for differential IK, and the resulting joint targets are applied to a robot
articulation supplied by a concrete calibrated executor.

The env exposes the duck-typed accessor surface the data engine consumes
(``door_frame_pose_w`` / ``hinge_state`` / ``ee_pose_w`` / ``set_ee_pose_w``)
plus optional
optional accessors (``robot_joint_state``, ``contact_force_w``,
``contact_sensed``) the engine picks up via ``hasattr``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from alexdoor_xas.assets.door_task import ensure_door_task_usd
from alexdoor_xas.kinematics import StartPoseError, check_settle_postcondition

from .door_contract import DOOR_PUSH_OBSERVATION_TERMS
from .door_runtime import read_doorframe_from_stage, resolve_hinge_joint_id
from .joint_limits import clamp_joint_targets

if TYPE_CHECKING:
    from .door_push_robot_env_cfg import DoorPushRobotEnvCfg


def _as_torch(value) -> torch.Tensor:
    """Unwrap this build's ``.torch`` data proxies; pass plain tensors through."""
    return value.torch if hasattr(value, "torch") else value


class DoorPushRobotEnv(DirectRLEnv):
    """Single-door task loop for a configured fixed-base articulated robot."""

    cfg: DoorPushRobotEnvCfg

    def __init__(self, cfg: DoorPushRobotEnvCfg, render_mode: str | None = None, **kwargs):
        usd_path = ensure_door_task_usd(cfg.door_pose_id)
        cfg.door_task_scene.spawn.usd_path = str(usd_path)
        if cfg.robot is None:
            raise RuntimeError("DoorPushRobotEnv requires an injected robot articulation config")
        if cfg.ee_contact is None:
            raise RuntimeError("DoorPushRobotEnv requires an injected EE contact sensor config")

        super().__init__(cfg, render_mode, **kwargs)

        self._hinge_joint_id = resolve_hinge_joint_id(
            list(self._door.joint_names), self.cfg.hinge_joint_name
        )
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._zero_reward = torch.zeros(self.num_envs, device=self.device)
        self._never_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._door_frame_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._door_frame_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        for env_id in self._all_env_ids.tolist():
            translation, quat_xyzw = read_doorframe_from_stage(env_id)
            self._door_frame_pos_w[env_id] = torch.tensor(
                translation, dtype=torch.float32, device=self.device
            )
            self._door_frame_quat_w[env_id] = torch.tensor(
                quat_xyzw, dtype=torch.float32, device=self.device
            )

        # -- IK plumbing (fixed base: jacobian body index shifts by -1, no
        # base-DoF column offset; see isaaclab task_space_actions).
        if not self._robot.is_fixed_base:
            raise RuntimeError("DoorPushRobotEnv requires a fixed-base robot articulation")
        ee_ids, _ = self._robot.find_bodies(self.cfg.ee_body_name)
        if len(ee_ids) != 1:
            raise RuntimeError(
                f"expected exactly one EE body {self.cfg.ee_body_name!r}, got {ee_ids}"
            )
        self._ee_body_idx = ee_ids[0]
        self._jacobi_body_idx = self._ee_body_idx - 1
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(
            list(self.cfg.arm_joint_names), preserve_order=True
        )
        # Position-only IK: the scripted task commands pure translations, and a
        # 6-DoF pose constraint is ill-conditioned from the arm's ready pose
        # (dls trades the tiny translation gain against large near-null joint
        # swings and the EE barely moves). Rotation action components are
        # clamped and recorded but not actuated — the same contract as the
        # earlier task execution, where rotation was physically inert.
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method=self.cfg.ik_method
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
        # Applied joint-position targets, tracked env-side so the recorded
        # proprio does not depend on backend target-readback support.
        self._joint_targets = _as_torch(self._robot.data.default_joint_pos).clone()
        self._arm_targets = self._joint_targets[:, self._arm_joint_ids].clone()
        # Anti-windup: solved IK targets are clamped to the arm's position
        # limits every solve; telemetry keeps the raw pre-clamp excess so the
        # data engine can report how hard the solver leaned on the limits.
        arm_pos_limits = _as_torch(self._robot.data.joint_pos_limits)[:, self._arm_joint_ids, :]
        self._arm_pos_lower = arm_pos_limits[..., 0].clone()
        self._arm_pos_upper = arm_pos_limits[..., 1].clone()
        n_arm = len(self._arm_joint_ids)
        self._clamp_excess_max = torch.zeros((self.num_envs, n_arm), device=self.device)
        self._clamp_count = torch.zeros(
            (self.num_envs, n_arm), dtype=torch.long, device=self.device
        )
        self._clamp_solve_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_settle_report: dict | None = None

    # -- scene ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        if self.cfg.door_task_scene.spawn is None:
            raise RuntimeError("DoorPushRobotEnvCfg.door_task_scene.spawn must be configured")

        self.cfg.door_task_scene.spawn.func(
            self.cfg.door_task_scene.prim_path,
            self.cfg.door_task_scene.spawn,
            translation=self.cfg.door_task_scene.init_state.pos,
            orientation=self.cfg.door_task_scene.init_state.rot,
        )
        self._door = Articulation(self.cfg.door)
        self._robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.ee_contact)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # A URDF importer may nest link rigid bodies below the articulation root,
        # while automatic activation stops at the first rigid body. Apply the
        # contact-report API to the configured EE link explicitly.
        from isaaclab.sim.schemas import activate_contact_sensors  # noqa: PLC0415

        contact_prim_pattern = str(self.cfg.ee_contact.prim_path)
        if "env_.*" not in contact_prim_pattern:
            raise RuntimeError(
                "EE contact sensor prim path must contain the environment token 'env_.*': "
                f"{contact_prim_pattern}"
            )
        for env_id in range(self.scene.cfg.num_envs):
            activate_contact_sensors(
                contact_prim_pattern.replace("env_.*", f"env_{env_id}", 1)
            )

        self.scene.articulations["door"] = self._door
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["ee_contact"] = self._contact_sensor

    # -- control ----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.cfg.action_space)
        if actions.shape != expected_shape:
            raise ValueError(
                f"DoorPushRobotEnv expects actions with shape {expected_shape}, "
                f"got {tuple(actions.shape)}"
            )
        clamped = actions.clone()
        clamped[:, :3] = clamped[:, :3].clamp(-self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
        clamped[:, 3:] = clamped[:, 3:].clamp(
            -self.cfg.max_rot_delta_rad, self.cfg.max_rot_delta_rad
        )
        self.actions = clamped
        self._solve_arm_ik(clamped)

    def _solve_arm_ik(self, delta_pose: torch.Tensor) -> None:
        """Turn a world-frame EE delta into right-arm joint targets.

        World frame throughout (pose, jacobian, delta are all world-frame),
        which is self-consistent and valid because the base is fixed. Only the
        translation components are commanded (position-mode IK; see __init__).
        """
        ee_pos, ee_quat = self._ee_pose_w()
        self._ik.set_command(delta_pose[:, :3], ee_pos=ee_pos, ee_quat=ee_quat)
        jacobian = _as_torch(self._robot.data.body_link_jacobian_w)[:, self._jacobi_body_idx][
            :, :, self._arm_joint_ids
        ]
        joint_pos = _as_torch(self._robot.data.joint_pos)[:, self._arm_joint_ids]
        raw_targets = self._ik.compute(ee_pos, ee_quat, jacobian, joint_pos)
        self._arm_targets, excess = clamp_joint_targets(
            raw_targets, self._arm_pos_lower, self._arm_pos_upper
        )
        self._clamp_excess_max = torch.maximum(self._clamp_excess_max, excess)
        self._clamp_count += (excess > 0).long()
        self._clamp_solve_ticks += 1
        self._joint_targets[:, self._arm_joint_ids] = self._arm_targets

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._arm_targets, joint_ids=self._arm_joint_ids, env_ids=self._all_env_ids
        )

    # -- MDP terms ---------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        hinge_pos = _as_torch(self._door.data.joint_pos)[:, self._hinge_joint_id].unsqueeze(-1)
        hinge_vel = _as_torch(self._door.data.joint_vel)[:, self._hinge_joint_id].unsqueeze(-1)
        ee_pos, ee_quat = self._ee_pose_w()
        obs = torch.cat((hinge_pos, hinge_vel, ee_pos - self.scene.env_origins, ee_quat), dim=-1)
        if not torch.isfinite(obs).all():
            raise RuntimeError(
                "door push observations "
                f"{DOOR_PUSH_OBSERVATION_TERMS} contain non-finite values: "
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

        # Door: hinge state only; the root is world-anchored (same rule as the
        # Never write root poses for world-anchored articulations.
        door_joint_pos = _as_torch(self._door.data.default_joint_pos)[env_ids].clone()
        door_joint_vel = torch.zeros_like(_as_torch(self._door.data.default_joint_vel)[env_ids])
        self._door.write_joint_position_to_sim_index(position=door_joint_pos, env_ids=env_ids)
        self._door.write_joint_velocity_to_sim_index(velocity=door_joint_vel, env_ids=env_ids)

        # Robot: default joint state + position targets (fixed base, no root write).
        joint_pos = _as_torch(self._robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(_as_torch(self._robot.data.default_joint_vel)[env_ids])
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

        if hasattr(self, "_joint_targets"):
            self._joint_targets[env_ids] = joint_pos
            self._arm_targets[env_ids] = joint_pos[:, self._arm_joint_ids]
            self._ik.reset(env_ids)
            self._clamp_excess_max[env_ids] = 0.0
            self._clamp_count[env_ids] = 0
            self._clamp_solve_ticks[env_ids] = 0
            self._last_settle_report = None

    # -- Phase 2 state accessors (used by the scripted controller / recorder) ----

    def door_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the ``Doorframe`` body (read from the USD stage at init)."""
        return self._door_frame_pos_w.clone(), self._door_frame_quat_w.clone()

    def hinge_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Hinge angle (rad) and angular velocity (rad/s), shape ``(num_envs,)``."""
        return (
            _as_torch(self._door.data.joint_pos)[:, self._hinge_joint_id].clone(),
            _as_torch(self._door.data.joint_vel)[:, self._hinge_joint_id].clone(),
        )

    def ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and ``(x, y, z, w)`` orientation of the configured EE."""
        return self._ee_pose_w()

    def set_ee_pose_w(
        self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Drive the EE toward a requested start position with a bounded IK settle.

        A robot EE can only move through its kinematics: this runs up to
        ``cfg.settle_ticks`` physics-only control
        ticks of the same clamped IK tracking the policy path uses. Orientation
        is ignored (the scripted task commands no rotation), so no orientation
        residual is defined. Fail-closed postcondition: the realized EE
        position must land within ``cfg.start_pose_tolerance_m`` of the request
        or :class:`StartPoseError` aborts the episode/rollout; the measured
        requested/realized/residual record is kept for provenance
        (:meth:`start_pose_settle_report`).
        """
        if env_ids is None:
            env_ids = self._all_env_ids
        goal = pos_w.to(device=self.device, dtype=torch.float32).reshape(len(env_ids), 3)
        del quat_w  # orientation intentionally not tracked

        ticks_used = 0
        for _ in range(self.cfg.settle_ticks):
            ee_pos, _ = self._ee_pose_w()
            error = goal - ee_pos[env_ids]
            if bool((error.norm(dim=-1) < self.cfg.settle_target_m).all()):
                break
            delta = torch.zeros((self.num_envs, 6), device=self.device)
            delta[env_ids, :3] = error.clamp(
                -self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m
            )
            self._solve_arm_ik(delta)
            for _ in range(self.cfg.decimation):
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
            ticks_used += 1

        ee_pos, _ = self._ee_pose_w()
        try:
            report = check_settle_postcondition(
                goal[0].detach().cpu().numpy(),
                ee_pos[env_ids][0].detach().cpu().numpy(),
                settle_ticks_used=ticks_used,
                max_settle_ticks=int(self.cfg.settle_ticks),
                tolerance_m=float(self.cfg.start_pose_tolerance_m),
            )
        except StartPoseError as error:
            self._last_settle_report = error.report.to_dict()
            raise
        self._last_settle_report = report.to_dict()

    def start_pose_settle_report(self) -> dict | None:
        """Requested/realized/residual record of the last ``set_ee_pose_w``.

        ``None`` when no start pose was requested since the last reset (the
        default fixed reset needs no settle).
        """
        return self._last_settle_report

    # -- Alex V2 accessors (force contact + joint proprio) -------------------------

    def contact_force_w(self) -> torch.Tensor:
        """Gripper<->door contact force (N), world frame, shape ``(num_envs, 3)``.

        Prefers the filtered per-partner force (door panel only); falls back to
        the net contact force if filtered reporting is unavailable.
        """
        force_matrix = self._contact_sensor.data.force_matrix_w
        if force_matrix is not None:
            return _as_torch(force_matrix)[:, 0, 0].clone()
        return _as_torch(self._contact_sensor.data.net_forces_w)[:, 0].clone()

    def contact_sensed(self, threshold_n: float | None = None) -> torch.Tensor:
        """Whether the sensed contact force exceeds the threshold, shape ``(num_envs,)``."""
        threshold = self.cfg.contact_force_threshold_n if threshold_n is None else threshold_n
        return self.contact_force_w().norm(dim=-1) >= threshold

    def robot_joint_state(self) -> dict:
        """Full-body joint state for episode proprio (numpy, env 0 layout ``(J,)``)."""
        joint_pos = _as_torch(self._robot.data.joint_pos).detach().cpu().numpy()[0]
        joint_vel = _as_torch(self._robot.data.joint_vel).detach().cpu().numpy()[0]
        targets = self._joint_targets.detach().cpu().numpy()[0]
        return {
            "joint_pos": joint_pos.copy(),
            "joint_vel": joint_vel.copy(),
            "joint_pos_target": targets.copy(),
        }

    def robot_joint_names(self) -> list[str]:
        return list(self._robot.joint_names)

    def robot_base_pos_w(self) -> torch.Tensor:
        """World position of the (fixed) pelvis base, shape ``(num_envs, 3)``.

        Reference point for the relative door-pose observation terms; live
        root reads are valid for the URDF-spawned Alex articulation.
        """
        return _as_torch(self._robot.data.root_pos_w).clone()

    def robot_joint_limits(self) -> dict:
        """Isaac-reported joint limits (numpy, env 0): position ``(J, 2)``, velocity ``(J,)``.

        Recorded into episode extras so the pure sanity checks
        (``eval/sanity.py``) can validate targets/velocities without Isaac.
        """
        pos_limits = _as_torch(self._robot.data.joint_pos_limits).detach().cpu().numpy()[0]
        vel_limits = _as_torch(self._robot.data.joint_vel_limits).detach().cpu().numpy()[0]
        return {
            "joint_pos_limits": pos_limits.copy(),
            "joint_vel_limits": vel_limits.copy(),
        }

    def arm_joint_ids(self) -> list[int]:
        return list(self._arm_joint_ids)

    def ik_clamp_telemetry(self) -> dict:
        """Anti-windup clamp telemetry since the last reset (env 0, JSON-able).

        Covers every IK solve after the reset — the ``set_ee_pose_w`` settle
        ticks and the per-control-tick episode solves — so a recorded episode
        reports exactly how often (and how far) the raw diff-IK targets ran
        past the arm's position limits before clamping.
        """
        excess = self._clamp_excess_max.detach().cpu().numpy()[0]
        counts = self._clamp_count.detach().cpu().numpy()[0]
        return {
            "joints": {
                name: {
                    "max_excess_rad": float(excess[i]),
                    "clamp_ticks": int(counts[i]),
                }
                for i, name in enumerate(self._arm_joint_names)
            },
            "n_solve_ticks": int(self._clamp_solve_ticks.detach().cpu().numpy()[0]),
            "max_excess_rad": float(excess.max()) if excess.size else 0.0,
            "clamp_ticks_total": int(counts.sum()),
        }

    # -- internals ----------------------------------------------------------------

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = _as_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        return pos.clone(), quat.clone()


__all__ = ["DoorPushRobotEnv"]
