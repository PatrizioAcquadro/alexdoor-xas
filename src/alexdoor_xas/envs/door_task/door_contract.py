"""Shared door-benchmark paths and action/observation contracts."""

from __future__ import annotations

DOOR_TASK_SCENE_SOURCE_PRIM_PATH = "/World/envs/env_0/DoorTaskScene"
DOOR_TASK_ARTICULATION_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene/DoorTaskDoor"
DOOR_TASK_SCENE_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene"

DOOR_PUSH_OBSERVATION_TERMS = (
    "door_angle_rad",
    "door_angular_velocity_rad_s",
    "ee_pos_x_m",
    "ee_pos_y_m",
    "ee_pos_z_m",
    "ee_quat_w",
    "ee_quat_x",
    "ee_quat_y",
    "ee_quat_z",
)
DOOR_PUSH_ACTION_TERMS = (
    "d_pos_x_m",
    "d_pos_y_m",
    "d_pos_z_m",
    "d_rot_x_rad",
    "d_rot_y_rad",
    "d_rot_z_rad",
)

__all__ = [
    "DOOR_PUSH_ACTION_TERMS",
    "DOOR_PUSH_OBSERVATION_TERMS",
    "DOOR_TASK_ARTICULATION_PRIM_PATH",
    "DOOR_TASK_SCENE_PRIM_PATH",
    "DOOR_TASK_SCENE_SOURCE_PRIM_PATH",
]
