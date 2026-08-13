"""Isaac Lab configuration for the Alex V2 door benchmark."""

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

from alexdoor_xas import paths

ALEX_V2_LIMITATIONS = (
    "Alex V2 is fixed-base: no stepping, balancing, or whole-body recovery is executed.",
    "Only the six right-arm joints are driven by position-only differential IK; "
    "rotation action components remain recorded but are not actuated.",
    "The controlled EE is the collision-derived gripper support point and its "
    "point Jacobian, both pinned to the validated runtime manifest.",
    "Contact force is selected by exact door actor ID from PhysX raw GPU contacts; "
    "unfiltered net force is never accepted.",
    "The standard articulated forearm collision union is used without external hands.",
)

_ALEX_V2_PRIM_PATH = "/World/envs/env_.*/Alex"
_ALEX_V2_EE_BODY_NAME = "RIGHT_GRIPPER_Z_LINK"
_ALEX_V2_SHOULDER_BODY_NAME = "RIGHT_SHOULDER_Z_LINK"
_ALEX_V2_EE_LINK_SUBPATH = (
    "Geometry/PELVIS_LINK/TORSO_LINK/RIGHT_SHOULDER_Y_LINK/RIGHT_SHOULDER_X_LINK/"
    "RIGHT_SHOULDER_Z_LINK/RIGHT_ELBOW_Y_LINK/RIGHT_WRIST_Z_LINK/RIGHT_WRIST_X_LINK/"
    "RIGHT_GRIPPER_Z_LINK"
)
_ALEX_V2_ARM_JOINT_NAMES = (
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
)
_DOOR_SCENE_SOURCE_PRIM_PATH = "/World/envs/env_0/DoorScene"
_DOOR_ARTICULATION_PRIM_PATH = "/World/envs/env_.*/DoorScene/Door"
_DOOR_PANEL_BODY_PRIM_PATH = f"{_DOOR_ARTICULATION_PRIM_PATH}/Door"
_HINGE_DAMPING_NM_S_PER_RAD = 4.0


@configclass
class DoorPushAlexV2EnvCfg(DirectRLEnvCfg):
    """Single-environment fixed-base Alex V2 door task."""

    decimation = 2
    episode_length_s = 10.0
    action_space = 6
    observation_space = 9
    state_space = 0

    sim: SimulationCfg = SimulationCfg(device="cuda:0", dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    viewer: ViewerCfg = ViewerCfg(eye=(1.5, 2.6, 2.0), lookat=(-0.7, 0.0, 0.9))

    door_scene = AssetBaseCfg(
        prim_path=_DOOR_SCENE_SOURCE_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )
    door = ArticulationCfg(
        prim_path=_DOOR_ARTICULATION_PRIM_PATH,
        spawn=None,
        actuators={
            "hinge_damper": ImplicitActuatorCfg(
                joint_names_expr=["Hinge"],
                stiffness=0.0,
                damping=_HINGE_DAMPING_NM_S_PER_RAD,
            )
        },
    )
    robot: ArticulationCfg | None = None
    ee_contact = ContactSensorCfg(
        prim_path=f"{_ALEX_V2_PRIM_PATH}/{_ALEX_V2_EE_LINK_SUBPATH}",
        filter_prim_paths_expr=[],
        update_period=0.0,
        max_contact_data_count_per_prim=16,
    )

    calibration_path: str = str(paths.ALEX_V2_CALIBRATION)
    contact_force_threshold_n = 1.0
    settle_ticks = 90
    settle_target_m = 0.005
    start_pose_tolerance_m = 0.01
    max_pos_delta_m = 0.02
    max_rot_delta_rad = 0.05
    door_pose_id: str = "D0"
