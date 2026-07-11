"""Robot-agnostic configuration for the articulated door-push task loop."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .door_env_cfg import (
    DOOR_TASK_ARTICULATION_PRIM_PATH,
    DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
)
from .door_push_env_cfg import ACTION_TERMS, OBSERVATION_TERMS

DOOR_PANEL_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene/DoorTaskDoor/Door/Cylinder_001"
"""Door panel collision-shape prim (the 25 kg slab's only collider), used as the
contact-sensor filter. Must be the *shape* prim, not the ``Door`` body: PhysX
filter globs prefix-match path strings, so the body path also captures the
sibling ``Doorframe`` collider ("expected 1, found 2") and the filtered
``force_matrix_w`` view silently fails to build (measured on 6.0.1)."""

HINGE_DAMPING_NM_S_PER_RAD = 4.0
"""Passive hinge damping for the articulated-robot door task."""


@configclass
class DoorPushRobotEnvCfg(DirectRLEnvCfg):
    """Direct-RL config shared by calibrated articulated-robot executors.

    Frozen Phase 2 numbers are shared with the proxy env: same sim dt /
    decimation, action/observation terms, per-tick clamps, and door cfg. As on
    the proxy sphere, the A2 rotation deltas are clamped and recorded but NOT
    actuated: the env runs position-only differential IK (a 6-DoF pose
    constraint is ill-conditioned from the arm's ready pose — see the env).
    """

    decimation = 2
    episode_length_s = 10.0
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

    # Camera framing the robot + door for --video runs.
    viewer: ViewerCfg = ViewerCfg(eye=(0.9, 1.8, 1.9), lookat=(-0.7, 0.0, 1.1))

    door_task_scene = AssetBaseCfg(
        prim_path=DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )

    door = ArticulationCfg(
        prim_path=DOOR_TASK_ARTICULATION_PRIM_PATH,
        spawn=None,
        # Passive hinge damper (zero stiffness, zero-velocity target): the raw
        # task USD hinge is frictionless, so a single arm tap sends the door
        # coasting to full open ahead of the pusher and the FSM never gets a
        # sustained push (measured on the gate). With damping the door moves
        # only while pushed — closer to a real door. Articulated env only; the proxy
        # env keeps the undamped hinge (frozen Phase 2 behavior).
        actuators={
            "hinge_damper": ImplicitActuatorCfg(
                joint_names_expr=["Hinge"],
                stiffness=0.0,
                damping=HINGE_DAMPING_NM_S_PER_RAD,
            )
        },
    )

    robot: ArticulationCfg | None = None
    """Concrete articulation injected by the calibrated executor."""

    ee_contact: ContactSensorCfg | None = None
    """Concrete EE contact sensor injected by the robot-specific config."""

    contact_force_threshold_n = 1.0
    """Force norm above which the env reports sensed contact."""

    ee_body_name = ""
    arm_joint_names: tuple[str, ...] = ()
    ik_method = "dls"
    """Damped least squares: robust near singularities."""

    settle_ticks = 90
    """Physics ticks the reset-time IK settle loop may use to reach a requested
    start EE pose (see ``DoorPushRobotEnv.set_proxy_pose``)."""

    hinge_joint_name = "Hinge"
    """Door revolute joint name exposed by the generated task USD."""

    max_pos_delta_m = 0.02
    """Per-component position delta clamp, meters per control tick."""

    max_rot_delta_rad = 0.05
    """Per-component axis-angle rotation delta clamp, radians per control tick."""

    door_yaw_rad: float = 0.0
    """Door-task pose variation: yaw about the hinge axis, authored into the
    generated scene USD at env construction (fixed for the process). The robot's
    fixed base stays put — the pose moves the door relative to the robot."""

    door_offset_xy: tuple[float, float] = (0.0, 0.0)
    """Door-task pose variation: world-frame XY translation in meters."""


__all__ = [
    "DOOR_PANEL_PRIM_PATH",
    "DoorPushRobotEnvCfg",
]
