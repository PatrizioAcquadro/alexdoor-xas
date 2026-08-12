"""Configuration for the Phase 2 door-push env with a dynamic, velocity-driven proxy EE."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .door_contract import (
    DOOR_PUSH_ACTION_TERMS as ACTION_TERMS,
)
from .door_contract import (
    DOOR_PUSH_OBSERVATION_TERMS as OBSERVATION_TERMS,
)
from .door_contract import (
    DOOR_TASK_ARTICULATION_PRIM_PATH,
    DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
)

PROXY_EE_PRIM_PATH = "/World/envs/env_.*/ProxyEE"
PROXY_EE_ROBOT_TAG = "proxy_ee_sphere_v0"
"""Episode-meta robot tag: Phase 2 uses a dynamic, gravity-free, velocity-driven
sphere proxy (never kinematic; see the ``proxy_ee`` cfg comment), not Alex.

The proxy has no joints, so A1 (joint deltas) is a documented placeholder; the
env's A2-shaped EE-delta interface is what a later Alex adapter must implement.
"""

PROXY_EE_RADIUS_M = 0.05


@configclass
class DoorPushEnvCfg(DirectRLEnvCfg):
    """Direct-RL config for the scripted door-push task (door + proxy sphere)."""

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

    door_task_scene = AssetBaseCfg(
        prim_path=DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )

    door = ArticulationCfg(
        prim_path=DOOR_TASK_ARTICULATION_PRIM_PATH,
        spawn=None,
        actuators={},
    )

    proxy_ee = RigidObjectCfg(
        prim_path=PROXY_EE_PRIM_PATH,
        # Dynamic, gravity-free sphere driven by per-step velocity commands. A
        # kinematic body is not used: kinematic-target pose writes make PhysX
        # sweep the body through the scene at teleport speed, which destabilizes
        # the door articulation; solver-resolved contact from a velocity-driven
        # dynamic body is well behaved and closer to a real end-effector.
        spawn=sim_utils.SphereCfg(
            radius=PROXY_EE_RADIUS_M,
            rigid_props=sim_utils.RigidBodyBaseCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionBaseCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.1)),
        ),
        # Collision-free start on the push side of the panel; the data engine may
        # re-place the proxy door-relative at reset (see DoorPushEnv.set_proxy_pose).
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    hinge_joint_name = "Hinge"
    """Door revolute joint name exposed by the generated task USD."""

    max_pos_delta_m = 0.02
    """Per-component position delta clamp, meters per control tick."""

    max_rot_delta_rad = 0.05
    """Per-component axis-angle rotation delta clamp, radians per control tick."""

    door_yaw_rad: float = 0.0
    """Door-task pose variation: yaw about the hinge axis, authored into the
    generated scene USD at env construction (fixed for the process)."""

    door_offset_xy: tuple[float, float] = (0.0, 0.0)
    """Door-task pose variation: world-frame XY translation in meters."""


__all__ = [
    "ACTION_TERMS",
    "OBSERVATION_TERMS",
    "PROXY_EE_PRIM_PATH",
    "PROXY_EE_RADIUS_M",
    "PROXY_EE_ROBOT_TAG",
    "DoorPushEnvCfg",
]
