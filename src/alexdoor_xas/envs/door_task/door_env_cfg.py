"""Configuration for the single-door Isaac Lab DirectRLEnv shell."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

OBSERVATION_TERMS = ("door_angle_rad", "door_angular_velocity_rad_s")
ACTION_TERMS = ("noop_debug_action",)

DOOR_TASK_SCENE_SOURCE_PRIM_PATH = "/World/envs/env_0/DoorTaskScene"
DOOR_TASK_ARTICULATION_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene/DoorTaskDoor"
DOOR_TASK_SCENE_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene"


@configclass
class DoorTaskEnvCfg(DirectRLEnvCfg):
    """Minimal direct-RL config for the pre-Phase-2 single-door task."""

    decimation = 2
    episode_length_s = 5.0
    action_space = len(ACTION_TERMS)
    observation_space = len(OBSERVATION_TERMS)
    state_space = 0

    sim: SimulationCfg = SimulationCfg(device="cpu", dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    door_task_scene = AssetBaseCfg(
        prim_path=DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )

    door = ArticulationCfg(
        prim_path=DOOR_TASK_ARTICULATION_PRIM_PATH,
        spawn=None,
        actuators={},
    )

    hinge_joint_name = "Hinge"
    """Door revolute joint name exposed by the generated task USD."""

    action_clip = 1.0
    """Absolute value used when storing the no-op debug action."""


__all__ = [
    "ACTION_TERMS",
    "DOOR_TASK_ARTICULATION_PRIM_PATH",
    "DOOR_TASK_SCENE_PRIM_PATH",
    "DOOR_TASK_SCENE_SOURCE_PRIM_PATH",
    "DoorTaskEnvCfg",
    "OBSERVATION_TERMS",
]
