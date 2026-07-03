"""Configuration for the Phase 2.5 door-push env with the IHMC Alex humanoid.

Alex stands at a fixed base (pelvis welded to the world) facing the door and
pushes it open with the RIGHT arm via differential IK. The action interface is
identical to the proxy env (6-dim A2 EE delta, same clamps), so the scripted
controller and data engine run unchanged; only the executor changed from a
velocity-driven sphere to joint-position IK tracking on the arm.

Asset notes (hard-won):
- The default fullbody URDF has NO arm collision geometry; the
  ``fullbody_fullcollisions`` variant is required so the gripper can actually
  touch the door. Its ``RIGHT_GRIPPER_Z_LINK`` carries a 0.05 m collision
  sphere — the same radius as the Phase 2 proxy sphere.
- The vendored actuators are ``DelayedPDActuatorCfg`` with randomly sampled
  0-2 step delays (nondeterministic) and RL-walking soft arm gains; they are
  replaced with ``IdealPDActuatorCfg`` groups, stiffened on the right arm for
  IK tracking (Franka-IK-style high PD).
"""

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

ALEX_PRIM_PATH = "/World/envs/env_.*/Alex"
ALEX_ROBOT_TAG = "alex_v1_fullbody_fixedbase_v0"
"""Episode-meta robot tag: fixed-base fullbody Alex, right arm driven by diff-IK."""

ALEX_URDF_VARIANT = "fullbody_fullcollisions"
ALEX_EE_BODY_NAME = "RIGHT_GRIPPER_Z_LINK"
"""IK/contact body: the arm's terminal link (Alex has no separate hand/palm link)."""

ALEX_EE_LINK_SUBPATH = (
    "Geometry/PELVIS_LINK/TORSO_LINK/RIGHT_SHOULDER_Y_LINK/RIGHT_SHOULDER_X_LINK/"
    "RIGHT_SHOULDER_Z_LINK/RIGHT_ELBOW_Y_LINK/RIGHT_WRIST_Z_LINK/RIGHT_WRIST_X_LINK/"
    "RIGHT_GRIPPER_Z_LINK"
)
"""Stage path of the EE link under the Alex root: the URDF importer nests link
prims by kinematic chain under a ``Geometry`` scope (verified on stage by
``scripts/verify_alex_ik_probe.py``)."""

ALEX_EE_RADIUS_M = 0.05
"""Radius of the gripper link's collision sphere in the fullCollisions URDF."""

ALEX_ARM_JOINT_NAMES = (
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
)
"""Joints the differential IK solves for. ``RIGHT_GRIPPER_Z`` is excluded: it is
a passive stub with no useful DoF for pushing; a PD hold keeps it at zero."""

# Standing pose: facing -X (toward the door's push face). The y offset lines
# the right arm (which hangs ~0.31 m to the robot's right, i.e. world +y after
# the yaw-pi turn) up with the push corridor at push_radius_frac=0.40, so the
# arm pushes straight forward and the torso stays clear of the waypoint
# corridor. Verified numerically by scripts/verify_alex_ik_probe.py
# (arc-reachability check against the measured shoulder position).
ALEX_BASE_POS = (-0.45, -0.38, 0.93)
ALEX_BASE_ROT_XYZW = (0.0, 0.0, 1.0, 0.0)  # yaw = pi

ALEX_READY_JOINT_POS: dict[str, float] = {
    "RIGHT_SHOULDER_Y": 0.3,
    "RIGHT_ELBOW_Y": -0.8,
}
"""Arm 'ready' pose: elbow bent with the upper arm swung slightly back, so the
hand sits at ~(-0.51, -0.10, 0.84) — clear of the door face at x=-0.665 (touch
would start at EE x <= -0.615) and away from the straight-arm z-singularity
that stalls the IK. Measured under the implicit actuators (hinge undisturbed,
max joint vel 0.11 rad/s after settling)."""

# High-PD gains for IK tracking on the right arm (the vendored RL-walking gains
# are far too soft: wrist 5, gripper 0). The stiffness/damping RATIO matters as
# much as the magnitude: per-tick tracking is ~(k/c)*dt, so k/c=40/s gives
# ~50% of the commanded joint delta per 1/60 s control tick (measured; damping
# 30 at stiffness 300 tracked only ~10%/tick). Torques stay far below the arm
# effort limits (160.7 / 70.5 / 25 N*m).
RIGHT_ARM_STIFFNESS = {
    ".*SHOULDER_Y": 600.0,
    ".*SHOULDER_X": 600.0,
    ".*SHOULDER_Z": 600.0,
    ".*ELBOW_Y": 600.0,
    ".*WRIST_Z": 150.0,
    ".*WRIST_X": 150.0,
    ".*GRIPPER_Z": 20.0,
}
RIGHT_ARM_DAMPING = {
    ".*SHOULDER_Y": 15.0,
    ".*SHOULDER_X": 15.0,
    ".*SHOULDER_Z": 15.0,
    ".*ELBOW_Y": 15.0,
    ".*WRIST_Z": 4.0,
    ".*WRIST_X": 4.0,
    ".*GRIPPER_Z": 1.0,
}

DOOR_PANEL_PRIM_PATH = "/World/envs/env_.*/DoorTaskScene/DoorTaskDoor/Door"
"""Door panel body prim (the 25 kg slab), used as the contact-sensor filter."""

HINGE_DAMPING_NM_S_PER_RAD = 4.0
"""Passive hinge damping for the Alex env's door (see the ``door`` cfg comment)."""


def default_joint_pos() -> dict[str, float]:
    """Ready-pose joint dict with a non-overlapping catch-all.

    Isaac Lab's regex resolution is strict (a joint matching two patterns is an
    error), so the catch-all excludes the explicitly posed joints via a
    negative lookahead instead of a plain ``.*``.
    """
    posed = dict(ALEX_READY_JOINT_POS)
    if not posed:
        return {".*": 0.0}
    exclude = "|".join(posed)
    posed[f"(?!(?:{exclude})$).*"] = 0.0
    return posed


def build_alex_articulation_cfg() -> ArticulationCfg:
    """Alex articulation for the door-push task: fixed base, deterministic PD.

    Deep-copies the vendored IHMC config via
    :func:`alexdoor_xas.assets.alex.load_alex_articulation_cfg`, then replaces
    every ``DelayedPDActuatorCfg`` (randomly sampled delays are
    nondeterministic) with ``ImplicitActuatorCfg`` groups and stiffens the
    right arm for IK tracking. Implicit (solver-integrated) drives are
    required: an explicit software PD (``IdealPDActuatorCfg``) at the high IK
    gains is numerically unstable on the low-inertia wrist joints (measured:
    wrist pegged at its velocity limit within 2 ticks). Call after
    ``AppLauncher``.
    """
    from isaaclab.actuators import ImplicitActuatorCfg

    from alexdoor_xas.assets.alex import load_alex_articulation_cfg

    cfg = load_alex_articulation_cfg(ALEX_URDF_VARIANT, fix_base=True)
    cfg = cfg.replace(prim_path=ALEX_PRIM_PATH)
    cfg.init_state.pos = ALEX_BASE_POS
    cfg.init_state.rot = ALEX_BASE_ROT_XYZW
    cfg.init_state.joint_pos = default_joint_pos()
    # The fullCollisions geometry overlaps between neighbors at the zero pose
    # and makes the standing robot thrash (probe measured 16 rad/s joint
    # velocities). Self-collision is safely disabled: only the right arm moves,
    # along a push corridor verified to clear the torso.
    cfg.spawn.articulation_props.enabled_self_collisions = False

    def implicit_pd(template, joint_names_expr, *, stiffness=None, damping=None):
        return ImplicitActuatorCfg(
            joint_names_expr=list(joint_names_expr),
            stiffness=dict(stiffness if stiffness is not None else template.stiffness),
            damping=dict(damping if damping is not None else template.damping),
            armature=dict(template.armature),
            velocity_limit_sim=dict(template.velocity_limit_sim),
            effort_limit_sim=dict(template.effort_limit_sim),
        )

    vendored = cfg.actuators
    arm_exprs = [name.removeprefix("RIGHT_") for name in ALEX_ARM_JOINT_NAMES] + ["GRIPPER_Z"]
    # Left arm keeps the vendored soft gains (it only holds pose), except the
    # gripper whose vendored 0/0 gains would leave the link flopping.
    left_arm_stiffness = dict(vendored["arms"].stiffness)
    left_arm_stiffness[".*GRIPPER_Z"] = RIGHT_ARM_STIFFNESS[".*GRIPPER_Z"]
    left_arm_damping = dict(vendored["arms"].damping)
    left_arm_damping[".*GRIPPER_Z"] = RIGHT_ARM_DAMPING[".*GRIPPER_Z"]

    cfg.actuators = {
        "legs": implicit_pd(vendored["legs"], vendored["legs"].joint_names_expr),
        "torso": implicit_pd(vendored["torso"], vendored["torso"].joint_names_expr),
        "left_arm": implicit_pd(
            vendored["arms"],
            [f"LEFT_{expr}" for expr in arm_exprs],
            stiffness=left_arm_stiffness,
            damping=left_arm_damping,
        ),
        "right_arm": implicit_pd(
            vendored["arms"],
            [f"RIGHT_{expr}" for expr in arm_exprs],
            stiffness=RIGHT_ARM_STIFFNESS,
            damping=RIGHT_ARM_DAMPING,
        ),
    }
    return cfg


@configclass
class DoorPushAlexEnvCfg(DirectRLEnvCfg):
    """Direct-RL config for the Alex door-push task (door + fixed-base Alex).

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

    # Camera framing Alex + door for --video runs.
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
        # only while pushed — closer to a real door. Alex env only; the proxy
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
    """Alex articulation; filled by the env at construction time via
    :func:`build_alex_articulation_cfg` (needs the vendored Isaac cfg import)."""

    ee_contact = ContactSensorCfg(
        prim_path=f"{ALEX_PRIM_PATH}/{ALEX_EE_LINK_SUBPATH}",
        # No filter_prim_paths_expr: PhysX cannot build the filtered pair view
        # against the referenced door USD (the panel pattern resolves to two
        # rigid-contact entries, and prefix matching also catches `Doorframe`).
        # The unfiltered net force is unambiguous here — the gripper can only
        # touch the door assembly in this scene.
        update_period=0.0,
        max_contact_data_count_per_prim=16,
    )
    """Gripper contact force sensor (force sensing, not geometric inference)."""

    contact_force_threshold_n = 1.0
    """Force norm above which the env reports sensed contact."""

    ee_body_name = ALEX_EE_BODY_NAME
    arm_joint_names = ALEX_ARM_JOINT_NAMES
    ik_method = "dls"
    """Damped least squares: robust near singularities."""

    settle_ticks = 90
    """Physics ticks the reset-time IK settle loop may use to reach a requested
    start EE pose (see ``DoorPushAlexEnv.set_proxy_pose``)."""

    hinge_joint_name = "Hinge"
    """Door revolute joint name exposed by the generated task USD."""

    max_pos_delta_m = 0.02
    """Per-component position delta clamp, meters per control tick."""

    max_rot_delta_rad = 0.05
    """Per-component axis-angle rotation delta clamp, radians per control tick."""


__all__ = [
    "ALEX_ARM_JOINT_NAMES",
    "ALEX_BASE_POS",
    "ALEX_BASE_ROT_XYZW",
    "ALEX_EE_BODY_NAME",
    "ALEX_EE_LINK_SUBPATH",
    "ALEX_EE_RADIUS_M",
    "ALEX_PRIM_PATH",
    "ALEX_READY_JOINT_POS",
    "ALEX_ROBOT_TAG",
    "ALEX_URDF_VARIANT",
    "DOOR_PANEL_PRIM_PATH",
    "RIGHT_ARM_DAMPING",
    "RIGHT_ARM_STIFFNESS",
    "DoorPushAlexEnvCfg",
    "build_alex_articulation_cfg",
    "default_joint_pos",
]
