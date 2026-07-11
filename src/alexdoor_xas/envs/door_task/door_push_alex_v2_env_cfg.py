"""Dedicated, fail-closed configuration for the Alex V2 door lineage."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from alexdoor_xas.calibration.alex_v2_door import default_calibration_path

from .alex_v2_runtime import ALEX_V2_PRIM_PATH
from .door_push_robot_env_cfg import (
    DOOR_PANEL_PRIM_PATH,
    DoorPushRobotEnvCfg,
)

ALEX_V2_ROBOT_TAG = "alex_v2_fullbody_fixedbase_standard_forearm_v0"
ALEX_V2_EE_BODY_NAME = "RIGHT_GRIPPER_Z_LINK"
ALEX_V2_SHOULDER_BODY_NAME = "RIGHT_SHOULDER_Z_LINK"
ALEX_V2_EE_LINK_SUBPATH = (
    "Geometry/PELVIS_LINK/TORSO_LINK/RIGHT_SHOULDER_Y_LINK/RIGHT_SHOULDER_X_LINK/"
    "RIGHT_SHOULDER_Z_LINK/RIGHT_ELBOW_Y_LINK/RIGHT_WRIST_Z_LINK/RIGHT_WRIST_X_LINK/"
    "RIGHT_GRIPPER_Z_LINK"
)
ALEX_V2_ARM_JOINT_NAMES = (
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
)


@configclass
class DoorPushAlexV2EnvCfg(DoorPushRobotEnvCfg):
    """V2 task config; production construction requires validated calibration."""

    calibration_path: str = str(default_calibration_path())
    ee_body_name = ALEX_V2_EE_BODY_NAME
    arm_joint_names = ALEX_V2_ARM_JOINT_NAMES
    ee_contact = ContactSensorCfg(
        prim_path=f"{ALEX_V2_PRIM_PATH}/{ALEX_V2_EE_LINK_SUBPATH}",
        filter_prim_paths_expr=[DOOR_PANEL_PRIM_PATH],
        update_period=0.0,
        max_contact_data_count_per_prim=16,
    )
    """V2 force sensor filtered to the exact door-panel collision shape.

    V2 execution requires ``force_matrix_w`` and never substitutes a net force
    containing contacts with other scene bodies.
    """


__all__ = [
    "ALEX_V2_ARM_JOINT_NAMES",
    "ALEX_V2_EE_BODY_NAME",
    "ALEX_V2_EE_LINK_SUBPATH",
    "ALEX_V2_ROBOT_TAG",
    "ALEX_V2_SHOULDER_BODY_NAME",
    "DoorPushAlexV2EnvCfg",
]
