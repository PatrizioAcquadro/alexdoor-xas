"""Shared collision-offset Alex V2 executor for production and calibration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch

from alexdoor_xas.assets.alex_v2 import load_alex_v2_articulation_cfg
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef
from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration
from alexdoor_xas.kinematics.offset_point import (
    compose_offset_pose_xyzw,
    link_jacobian_to_point,
)

from .alex_v2_runtime import inject_alex_v2_runtime_cfg
from .door_push_robot_env import DoorPushRobotEnv
from .joint_limits import clamp_joint_targets

if TYPE_CHECKING:
    from .door_push_alex_v2_env_cfg import DoorPushAlexV2EnvCfg


def _as_torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


class DoorPushAlexV2Executor(DoorPushRobotEnv):
    """V2-only execution semantics layered over the mature door task loop.

    The inherited scene/reset/observation surface remains useful, while robot
    configuration, end-effector pose, differential IK Jacobian, and contact
    force semantics are supplied here. A dedicated V2 config is injected before
    ``super()``, so no legacy asset or actuator fallback is possible.
    """

    cfg: DoorPushAlexV2EnvCfg
    candidate_only = False

    def __init__(
        self,
        cfg: DoorPushAlexV2EnvCfg,
        *,
        calibration: AlexV2DoorCalibration,
        runtime_asset: RobotAssetRef,
        runtime_manifest: Mapping[str, Any],
        render_mode: str | None = None,
        **kwargs,
    ):
        self._alex_v2_calibration = calibration
        self._alex_v2_robot_asset = runtime_asset
        self._alex_v2_runtime_manifest = deepcopy(dict(runtime_manifest))
        self._tool_translation_link = tuple(
            float(value) for value in calibration.tool_frame["translation_m"]
        )
        self._tool_orientation_link_xyzw = tuple(
            float(value) for value in calibration.tool_frame["orientation_xyzw"]
        )
        robot_cfg = load_alex_v2_articulation_cfg(fix_base=True)
        inject_alex_v2_runtime_cfg(cfg, robot_cfg, calibration)
        super().__init__(cfg, render_mode, **kwargs)
        from .door_push_alex_v2_env_cfg import ALEX_V2_SHOULDER_BODY_NAME  # noqa: PLC0415

        shoulder_ids, _ = self._robot.find_bodies(ALEX_V2_SHOULDER_BODY_NAME)
        if len(shoulder_ids) != 1:
            raise RuntimeError(
                f"expected exactly one shoulder body {ALEX_V2_SHOULDER_BODY_NAME!r}, "
                f"got {shoulder_ids}"
            )
        self._shoulder_body_idx = shoulder_ids[0]

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collision-derived push-tool pose, not the gripper link origin."""

        link_pos = _as_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        link_quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        translation = link_pos.new_tensor(self._tool_translation_link)
        orientation = link_quat.new_tensor(self._tool_orientation_link_xyzw)
        tool_pos, tool_quat = compose_offset_pose_xyzw(
            link_pos,
            link_quat,
            translation,
            orientation,
        )
        return tool_pos.clone(), tool_quat.clone()

    def _solve_arm_ik(self, delta_pose: torch.Tensor) -> None:
        """Solve translation IK with the Jacobian shifted to the push point."""

        tool_pos, tool_quat = self._ee_pose_w()
        self._ik.set_command(delta_pose[:, :3], ee_pos=tool_pos, ee_quat=tool_quat)

        point_jacobian = self.point_jacobian_w()

        joint_pos = _as_torch(self._robot.data.joint_pos)[:, self._arm_joint_ids]
        raw_targets = self._ik.compute(
            tool_pos,
            tool_quat,
            point_jacobian,
            joint_pos,
        )
        self._arm_targets, excess = clamp_joint_targets(
            raw_targets,
            self._arm_pos_lower,
            self._arm_pos_upper,
        )
        self._clamp_excess_max = torch.maximum(self._clamp_excess_max, excess)
        self._clamp_count += (excess > 0).long()
        self._clamp_solve_ticks += 1
        self._joint_targets[:, self._arm_joint_ids] = self._arm_targets

    def contact_force_w(self) -> torch.Tensor:
        """Sum every exact-door filter entry; never use unfiltered net force."""

        force_matrix = self._contact_sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(
                "Alex V2 requires filtered force_matrix_w for the exact door panel"
            )
        force = _as_torch(force_matrix)
        if force.ndim != 4 or force.shape[-1] != 3:
            raise RuntimeError(
                "Alex V2 filtered contact force must have shape (N, B, F, 3), "
                f"got {tuple(force.shape)}"
            )
        if force.shape[1] < 1 or force.shape[2] < 1:
            raise RuntimeError("Alex V2 filtered contact force contains no filter entries")
        if not bool(torch.isfinite(force).all()):
            raise RuntimeError("Alex V2 filtered contact force contains non-finite values")
        return force.sum(dim=(1, 2)).clone()

    def point_jacobian_w(self) -> torch.Tensor:
        """World-frame tool-point Jacobian for the six ordered arm joints."""

        link_quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        link_jacobian = _as_torch(self._robot.data.body_link_jacobian_w)[
            :, self._jacobi_body_idx
        ][:, :, self._arm_joint_ids]
        translation = link_quat.new_tensor(self._tool_translation_link)
        point_jacobian = link_jacobian_to_point(
            link_jacobian,
            link_quat,
            translation,
        )
        expected = (self.num_envs, 6, len(self._arm_joint_ids))
        if tuple(point_jacobian.shape) != expected:
            raise RuntimeError(
                f"Alex V2 point Jacobian must have shape {expected}, "
                f"got {tuple(point_jacobian.shape)}"
            )
        if not bool(torch.isfinite(point_jacobian).all()):
            raise RuntimeError("Alex V2 collision-offset point Jacobian is non-finite")
        return point_jacobian.clone()

    def alex_v2_calibration(self) -> AlexV2DoorCalibration:
        """Validated/candidate calibration selected by the concrete boundary."""

        return self._alex_v2_calibration

    def shoulder_position_world_m(self) -> torch.Tensor:
        """Live V2 shoulder center used with the calibrated reach shell."""

        position = _as_torch(self._robot.data.body_link_pos_w)[:, self._shoulder_body_idx]
        if not bool(torch.isfinite(position).all()):
            raise RuntimeError("Alex V2 shoulder position contains non-finite values")
        return position.clone()

    def robot_asset_provenance(self) -> dict[str, Any]:
        """Pure JSON-able identity consumed additively by the data engine."""

        return {
            **self._alex_v2_robot_asset.to_dict(),
            "manifest": deepcopy(self._alex_v2_runtime_manifest),
        }

    def ee_contact_prim_path(self, env_id: int = 0) -> str:
        """Resolved gripper contact-sensor prim for one cloned environment."""

        if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
            raise ValueError("env_id must be a non-negative integer")
        pattern = str(self.cfg.ee_contact.prim_path)
        token = "env_.*"
        if token not in pattern:
            raise RuntimeError(
                f"Alex V2 contact sensor path lacks the environment token: {pattern}"
            )
        return pattern.replace(token, f"env_{env_id}", 1)


__all__ = ["DoorPushAlexV2Executor"]
