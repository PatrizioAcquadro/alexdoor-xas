"""Calibrated single-environment Alex V2 door benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms

from alexdoor_xas.assets.alex_v2 import build_alex_v2_door_asset, load_alex_v2_articulation_cfg
from alexdoor_xas.assets.alex_v2_tool_frame import derive_right_gripper_tool_frame
from alexdoor_xas.assets.door_task import ensure_door_task_usd
from alexdoor_xas.calibration.alex_v2_door import (
    AlexV2DoorCalibration,
    load_alex_v2_door_calibration,
)
from alexdoor_xas.kinematics.point_jacobian import link_jacobian_to_point
from alexdoor_xas.kinematics.settle import StartPoseError, validate_start_pose_settle

from .contact_force import sum_actor_contact_force
from .door_push_alex_v2_env_cfg import (
    _ALEX_V2_ARM_JOINT_NAMES,
    _ALEX_V2_EE_BODY_NAME,
    _ALEX_V2_PRIM_PATH,
    _ALEX_V2_SHOULDER_BODY_NAME,
    _DOOR_PANEL_BODY_PRIM_PATH,
    DoorPushAlexV2EnvCfg,
)


def _as_torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _configure_robot(
    cfg: DoorPushAlexV2EnvCfg,
    calibration: AlexV2DoorCalibration,
) -> None:
    robot_cfg = load_alex_v2_articulation_cfg().replace(prim_path=_ALEX_V2_PRIM_PATH)
    robot_cfg.init_state.pos = tuple(calibration.base_pose["position_m"])
    robot_cfg.init_state.rot = tuple(calibration.base_pose["orientation_xyzw"])
    ready = {name: float(value) for name, value in calibration.ready_joint_pos.items()}
    ready[f"(?!(?:{'|'.join(ready)})$).*"] = 0.0
    robot_cfg.init_state.joint_pos = ready
    cfg.robot = robot_cfg
    cfg.contact_force_threshold_n = float(calibration.controller["contact_force_threshold_n"])


def _validate_tool_frame(
    manifest: Mapping[str, Any],
    calibration: AlexV2DoorCalibration,
) -> None:
    normal = calibration.tool_frame["contact_normal_link"]
    if derive_right_gripper_tool_frame(manifest, normal).to_dict() != dict(calibration.tool_frame):
        raise RuntimeError("calibrated tool frame differs from the current collision manifest")


def _read_door_frame_from_stage() -> tuple[tuple[float, ...], tuple[float, ...]]:
    import omni.usd  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    prim_path = "/World/envs/env_0/DoorTaskScene/DoorTaskDoor/Doorframe"
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"door frame prim not found on stage: {prim_path}")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    quat = transform.RemoveScaleShear().ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        (translation[0], translation[1], translation[2]),
        (imaginary[0], imaginary[1], imaginary[2], quat.GetReal()),
    )


class DoorPushAlexV2Env(DirectRLEnv):
    """Position-only right-arm control of the calibrated Alex V2 door task."""

    cfg: DoorPushAlexV2EnvCfg

    def __init__(
        self,
        cfg: DoorPushAlexV2EnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        if cfg.scene.num_envs != 1:
            raise ValueError("DoorPushAlexV2Env supports exactly one environment")

        asset, runtime_asset = build_alex_v2_door_asset()
        calibration = load_alex_v2_door_calibration(cfg.calibration_path)
        _validate_tool_frame(asset.manifest, calibration)
        _configure_robot(cfg, calibration)
        cfg.door_task_scene.spawn.usd_path = str(ensure_door_task_usd(cfg.door_pose_id))

        self._calibration = calibration
        self._robot_asset = runtime_asset
        self._runtime_manifest = deepcopy(asset.manifest)
        self._door_contact_actor_id: int | None = None

        super().__init__(cfg, render_mode, **kwargs)

        self._tool_translation_link = torch.tensor(
            [calibration.tool_frame["translation_m"]], dtype=torch.float32, device=self.device
        )
        self._tool_orientation_link_xyzw = torch.tensor(
            [calibration.tool_frame["orientation_xyzw"]],
            dtype=torch.float32,
            device=self.device,
        )

        hinge_matches = [
            index
            for index, name in enumerate(self._door.joint_names)
            if name.lower() == "hinge" or name.rsplit("/", 1)[-1].lower() == "hinge"
        ]
        if len(hinge_matches) != 1:
            raise RuntimeError(f"expected one door hinge, got {self._door.joint_names}")
        self._hinge_joint_id = hinge_matches[0]

        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.actions = torch.zeros((1, self.cfg.action_space), device=self.device)
        self._zero_reward = torch.zeros(1, device=self.device)
        self._never_done = torch.zeros(1, dtype=torch.bool, device=self.device)
        translation, quat_xyzw = _read_door_frame_from_stage()
        self._door_frame_pos_w = torch.tensor(
            [translation], dtype=torch.float32, device=self.device
        )
        self._door_frame_quat_w = torch.tensor([quat_xyzw], dtype=torch.float32, device=self.device)

        if not self._robot.is_fixed_base:
            raise RuntimeError("Alex V2 door execution requires a fixed-base robot")
        ee_ids, _ = self._robot.find_bodies(_ALEX_V2_EE_BODY_NAME)
        shoulder_ids, _ = self._robot.find_bodies(_ALEX_V2_SHOULDER_BODY_NAME)
        if len(ee_ids) != 1 or len(shoulder_ids) != 1:
            raise RuntimeError("Alex V2 EE and shoulder bodies must resolve uniquely")
        self._ee_body_idx = ee_ids[0]
        self._shoulder_body_idx = shoulder_ids[0]

        # Fixed-base Jacobians omit the root body and base-DoF columns.
        self._jacobi_body_idx = self._ee_body_idx - 1
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(
            list(_ALEX_V2_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(self._arm_joint_names) != _ALEX_V2_ARM_JOINT_NAMES or len(
            set(self._arm_joint_ids)
        ) != len(_ALEX_V2_ARM_JOINT_NAMES):
            raise RuntimeError("Alex V2 right-arm joints must resolve uniquely and in order")
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls"
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=1, device=self.device)
        self._joint_targets = _as_torch(self._robot.data.default_joint_pos).clone()
        self._arm_targets = self._joint_targets[:, self._arm_joint_ids].clone()
        arm_limits = _as_torch(self._robot.data.joint_pos_limits)[:, self._arm_joint_ids]
        self._arm_pos_lower = arm_limits[..., 0].clone()
        self._arm_pos_upper = arm_limits[..., 1].clone()
        if bool((self._arm_pos_lower > self._arm_pos_upper).any()):
            raise RuntimeError("Alex V2 arm position limits are invalid")

        n_arm = len(self._arm_joint_ids)
        self._clamp_excess_max = torch.zeros((1, n_arm), device=self.device)
        self._clamp_count = torch.zeros((1, n_arm), dtype=torch.long, device=self.device)
        self._clamp_solve_ticks = torch.zeros(1, dtype=torch.long, device=self.device)
        self._last_settle_report: dict | None = None

    def _setup_scene(self) -> None:
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

        from isaaclab.sim.schemas import activate_contact_sensors  # noqa: PLC0415

        contact_path = str(self.cfg.ee_contact.prim_path).replace("env_.*", "env_0", 1)
        activate_contact_sensors(contact_path)
        self.scene.articulations["door"] = self._door
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["ee_contact"] = self._contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        expected_shape = (1, self.cfg.action_space)
        if actions.shape != expected_shape:
            raise ValueError(f"expected actions with shape {expected_shape}, got {actions.shape}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("door push action contains non-finite values")
        self.actions = actions.clone()
        self.actions[:, :3].clamp_(
            -self.cfg.max_pos_delta_m,
            self.cfg.max_pos_delta_m,
        )
        self.actions[:, 3:].clamp_(
            -self.cfg.max_rot_delta_rad,
            self.cfg.max_rot_delta_rad,
        )
        self._solve_arm_ik(self.actions)

    def _solve_arm_ik(self, delta_pose: torch.Tensor) -> None:
        tool_pos, tool_quat = self._ee_pose_w()
        self._ik.set_command(delta_pose[:, :3], ee_pos=tool_pos, ee_quat=tool_quat)
        joint_pos = _as_torch(self._robot.data.joint_pos)[:, self._arm_joint_ids]
        raw_targets = self._ik.compute(
            tool_pos,
            tool_quat,
            self._point_jacobian_w(),
            joint_pos,
        )
        self._arm_targets = torch.clamp(
            raw_targets,
            min=self._arm_pos_lower,
            max=self._arm_pos_upper,
        )
        excess = (raw_targets - self._arm_targets).abs()
        self._clamp_excess_max = torch.maximum(self._clamp_excess_max, excess)
        self._clamp_count += (excess > 0).long()
        self._clamp_solve_ticks += 1
        self._joint_targets[:, self._arm_joint_ids] = self._arm_targets

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._arm_targets,
            joint_ids=self._arm_joint_ids,
            env_ids=self._all_env_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        hinge_pos = _as_torch(self._door.data.joint_pos)[:, self._hinge_joint_id].unsqueeze(-1)
        hinge_vel = _as_torch(self._door.data.joint_vel)[:, self._hinge_joint_id].unsqueeze(-1)
        ee_pos, ee_quat = self._ee_pose_w()
        obs = torch.cat((hinge_pos, hinge_vel, ee_pos - self.scene.env_origins, ee_quat), dim=-1)
        if not bool(torch.isfinite(obs).all()):
            raise RuntimeError("door push observation contains non-finite values")
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return self._zero_reward.clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._never_done.clone(), self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        env_ids = (
            self._all_env_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        )
        super()._reset_idx(env_ids)

        # Both articulations are world-anchored; reset joints without writing roots.
        door_pos = _as_torch(self._door.data.default_joint_pos)[env_ids].clone()
        door_vel = torch.zeros_like(_as_torch(self._door.data.default_joint_vel)[env_ids])
        self._door.write_joint_position_to_sim_index(position=door_pos, env_ids=env_ids)
        self._door.write_joint_velocity_to_sim_index(velocity=door_vel, env_ids=env_ids)

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

    def door_frame_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._door_frame_pos_w.clone(), self._door_frame_quat_w.clone()

    def hinge_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _as_torch(self._door.data.joint_pos)[:, self._hinge_joint_id].clone(),
            _as_torch(self._door.data.joint_vel)[:, self._hinge_joint_id].clone(),
        )

    def ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._ee_pose_w()

    def set_ee_pose_w(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        goal = pos_w.to(device=self.device, dtype=torch.float32).reshape(1, 3)
        del quat_w  # Position-only IK does not track orientation.

        ticks_used = 0
        for _ in range(self.cfg.settle_ticks):
            ee_pos, _ = self._ee_pose_w()
            error = goal - ee_pos
            if bool((error.norm(dim=-1) < self.cfg.settle_target_m).all()):
                break
            delta = torch.zeros((1, 6), device=self.device)
            delta[:, :3] = error.clamp(-self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
            self._solve_arm_ik(delta)
            for _ in range(self.cfg.decimation):
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
            ticks_used += 1

        realized, _ = self._ee_pose_w()
        try:
            report = validate_start_pose_settle(
                goal[0].detach().cpu().numpy(),
                realized[0].detach().cpu().numpy(),
                settle_ticks_used=ticks_used,
                max_settle_ticks=int(self.cfg.settle_ticks),
                tolerance_m=float(self.cfg.start_pose_tolerance_m),
            )
        except StartPoseError as error:
            self._last_settle_report = error.report.to_dict()
            raise
        self._last_settle_report = report.to_dict()

    def start_pose_settle_report(self) -> dict | None:
        return self._last_settle_report

    def contact_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        force_w = self._contact_force_w()
        sensed = force_w.norm(dim=-1) >= self.cfg.contact_force_threshold_n
        return force_w, sensed

    def _contact_force_w(self) -> torch.Tensor:
        contact_view = self._contact_sensor.contact_view
        if not hasattr(contact_view, "get_raw_contact_data"):
            raise RuntimeError("PhysX raw contact data is required for door-force sensing")
        (
            force_magnitudes,
            _,
            normals_w,
            _,
            counts,
            start_indices,
            other_actor_ids,
        ) = contact_view.get_raw_contact_data(dt=self.physics_dt)
        self._resolve_door_contact_actor_id(
            counts,
            start_indices,
            other_actor_ids,
            contact_view,
        )
        try:
            return sum_actor_contact_force(
                wp.to_torch(force_magnitudes),
                wp.to_torch(normals_w),
                wp.to_torch(counts),
                wp.to_torch(start_indices),
                wp.to_torch(other_actor_ids),
                self._door_contact_actor_id,
            ).clone()
        except ValueError as error:
            raise RuntimeError(f"invalid PhysX raw contact data: {error}") from error

    def _resolve_door_contact_actor_id(
        self,
        counts: Any,
        start_indices: Any,
        other_actor_ids: Any,
        contact_view: Any,
    ) -> None:
        if self._door_contact_actor_id is not None:
            return
        counts_cpu = np.asarray(counts.numpy(), dtype=np.int64).reshape(-1)
        starts_cpu = np.asarray(start_indices.numpy(), dtype=np.int64).reshape(-1)
        actor_ids_cpu = np.asarray(other_actor_ids.numpy(), dtype=np.uint64).reshape(-1)
        if counts_cpu.shape != (1,) or starts_cpu.shape != (1,):
            raise RuntimeError("raw contact data must describe one sensor")
        count = int(counts_cpu[0])
        start = int(starts_cpu[0])
        if count < 0 or start < 0 or start + count > actor_ids_cpu.size:
            raise RuntimeError("raw contact range is invalid")
        if count == 0:
            return

        unique_ids = np.unique(actor_ids_cpu[start : start + count])
        ids_buffer = wp.array(unique_ids, dtype=wp.uint64, device="cpu")
        paths = contact_view.get_other_actor_paths_from_ids(ids_buffer)
        expected = _DOOR_PANEL_BODY_PRIM_PATH.replace("env_.*", "env_0", 1)
        for actor_id, path in zip(unique_ids, paths, strict=True):
            path = str(path)
            if path == expected or path.startswith(expected + "/"):
                self._door_contact_actor_id = int(actor_id)
                return

    def alex_v2_calibration(self) -> AlexV2DoorCalibration:
        return self._calibration

    def shoulder_position_world_m(self) -> torch.Tensor:
        position = _as_torch(self._robot.data.body_link_pos_w)[:, self._shoulder_body_idx]
        if not bool(torch.isfinite(position).all()):
            raise RuntimeError("Alex V2 shoulder position contains non-finite values")
        return position.clone()

    def robot_asset_provenance(self) -> dict[str, Any]:
        return {**self._robot_asset.to_dict(), "manifest": deepcopy(self._runtime_manifest)}

    def robot_joint_state(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": _as_torch(self._robot.data.joint_pos).detach().cpu().numpy()[0].copy(),
            "joint_vel": _as_torch(self._robot.data.joint_vel).detach().cpu().numpy()[0].copy(),
            "joint_pos_target": self._joint_targets.detach().cpu().numpy()[0].copy(),
        }

    def robot_joint_names(self) -> list[str]:
        return list(self._robot.joint_names)

    def robot_base_pos_w(self) -> torch.Tensor:
        return _as_torch(self._robot.data.root_pos_w).clone()

    def robot_joint_limits(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos_limits": _as_torch(self._robot.data.joint_pos_limits)
            .detach()
            .cpu()
            .numpy()[0]
            .copy(),
            "joint_vel_limits": _as_torch(self._robot.data.joint_vel_limits)
            .detach()
            .cpu()
            .numpy()[0]
            .copy(),
        }

    def arm_joint_ids(self) -> list[int]:
        return list(self._arm_joint_ids)

    def ik_clamp_telemetry(self) -> dict[str, Any]:
        excess = self._clamp_excess_max.detach().cpu().numpy()[0]
        counts = self._clamp_count.detach().cpu().numpy()[0]
        return {
            "joints": {
                name: {
                    "max_excess_rad": float(excess[index]),
                    "clamp_ticks": int(counts[index]),
                }
                for index, name in enumerate(self._arm_joint_names)
            },
            "n_solve_ticks": int(self._clamp_solve_ticks.item()),
            "max_excess_rad": float(excess.max()) if excess.size else 0.0,
            "clamp_ticks_total": int(counts.sum()),
        }

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos = _as_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        link_quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        tool_pos, tool_quat = combine_frame_transforms(
            link_pos,
            link_quat,
            self._tool_translation_link,
            self._tool_orientation_link_xyzw,
        )
        if tool_pos.shape != (1, 3) or tool_quat.shape != (1, 4):
            raise RuntimeError("Alex V2 tool pose must have shapes (1, 3) and (1, 4)")
        if not bool(torch.isfinite(tool_pos).all() & torch.isfinite(tool_quat).all()):
            raise RuntimeError("Alex V2 tool pose must be finite")
        return tool_pos, tool_quat

    def _point_jacobian_w(self) -> torch.Tensor:
        link_quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        link_jacobian = _as_torch(self._robot.data.body_link_jacobian_w)[:, self._jacobi_body_idx][
            :, :, self._arm_joint_ids
        ]
        point_jacobian = link_jacobian_to_point(
            link_jacobian,
            link_quat,
            self._tool_translation_link,
        )
        expected = (1, 6, len(self._arm_joint_ids))
        if point_jacobian.shape != expected or not bool(torch.isfinite(point_jacobian).all()):
            raise RuntimeError(f"Alex V2 point Jacobian must be finite with shape {expected}")
        return point_jacobian
