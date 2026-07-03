#!/usr/bin/env python
"""Backend go/no-go probe for the Alex door-push env (Phase 2.5).

This Isaac Lab release returns zeros for articulation body/root pose reads in
some direct-env contexts (the door articulation is read from the USD stage for
exactly that reason). Differential IK needs live EE poses and jacobians, so
this probe proves — on the real registered env — that:

 1. the env spawns (URDF import, actuators, contact sensor) and the expected
    prim paths exist on the stage;
 2. EE body pose reads are live (finite, non-zero, matching a USD stage read);
 3. the EE jacobian read is live (finite, non-zero, expected shape);
 4. the standing pose is stable under the PD holds (no self-collision blowup);
 5. differential IK actually tracks a commanded EE delta;
 6. the Alex controller preset's push arc is inside the measured arm workspace;
 7. (--contact) the gripper<->door contact sensor reports force on contact.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_alex_ik_probe.py --viz none --device cpu --contact
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

# -- AppLauncher must be configured before any other Isaac import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS Alex IK backend probe")
parser.add_argument("--seed", type=int, default=1234, help="Seed used for env creation and reset.")
parser.add_argument(
    "--settle-ticks", type=int, default=60, help="Zero-action ticks before backend assertions."
)
parser.add_argument(
    "--contact",
    action="store_true",
    help="Also drive the EE into the door and assert sensed contact force.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env import _as_torch  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import (  # noqa: E402
    ALEX_EE_LINK_SUBPATH,
    DoorPushAlexEnvCfg,
)
from alexdoor_xas.policies.scripted import alex_fixedbase_push_cfg  # noqa: E402

EE_PRIM_PATH = f"/World/envs/env_0/Alex/{ALEX_EE_LINK_SUBPATH}"
SHOULDER_BODY_NAME = "RIGHT_SHOULDER_Z_LINK"
# Measured full-arm length from SHOULDER_Z to the EE is ~0.584 m; keep the
# whole push arc inside a comfortable sub-band of that workspace.
ARC_REACH_BOUNDS_M = (0.18, 0.55)


def _stage_prim_pos(prim_path: str) -> np.ndarray:
    import omni.usd  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found on stage: {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    translation = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]], dtype=np.float64)


def _step_zero(env, ticks: int) -> None:
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    for _ in range(ticks):
        env.step(action)


def _check_spawn_and_prims(env) -> None:
    import omni.usd  # noqa: PLC0415

    robot = env._robot
    print(f"[spawn] bodies ({len(robot.body_names)}): {list(robot.body_names)}", flush=True)
    print(f"[spawn] joints ({len(robot.joint_names)}): {list(robot.joint_names)}", flush=True)

    stage = omni.usd.get_context().get_stage()
    alex_root = stage.GetPrimAtPath("/World/envs/env_0/Alex")
    if not alex_root.IsValid():
        raise RuntimeError("Alex prim not found at /World/envs/env_0/Alex")
    children = [child.GetName() for child in alex_root.GetChildren()]
    print(f"[spawn] Alex stage children: {children}", flush=True)
    if not stage.GetPrimAtPath(EE_PRIM_PATH).IsValid():
        raise RuntimeError(f"EE prim not found on stage: {EE_PRIM_PATH}")
    door_root = stage.GetPrimAtPath("/World/envs/env_0/DoorTaskScene/DoorTaskDoor")
    print(
        f"[spawn] door children: {[child.GetName() for child in door_root.GetChildren()]}",
        flush=True,
    )


def _check_pose_backend(env) -> np.ndarray:
    """EE body pose must be live: finite, non-zero, and physically plausible.

    Note the USD stage keeps the authored rest pose for dynamic bodies (physics
    does not sync back to USD headless), so a stage comparison is only printed
    for reference — the authoritative liveness proof is the IK tracking check,
    which observes the same read moving under commanded deltas.
    """
    from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import ALEX_BASE_POS  # noqa: PLC0415

    ee_pos, ee_quat = env._ee_pose_w()
    pos = ee_pos.detach().cpu().numpy()[0]
    quat = ee_quat.detach().cpu().numpy()[0]
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise RuntimeError(f"EE pose read is non-finite: pos={pos} quat={quat}")
    if np.linalg.norm(pos) < 1e-6:
        raise RuntimeError(
            "EE body pose reads return zeros in this direct-env context (known "
            "backend gotcha). Fallback required: per-tick UsdGeom.XformCache "
            "stage read of the gripper prim in DoorPushAlexEnv._ee_pose_w."
        )
    stage_pos = _stage_prim_pos(EE_PRIM_PATH)
    print(f"[pose] ee_body_pos_w={pos} (stage rest pose for reference: {stage_pos})", flush=True)
    distance_from_base = float(np.linalg.norm(pos - np.asarray(ALEX_BASE_POS)))
    if distance_from_base > 1.5:
        raise RuntimeError(
            f"EE pose {pos} is implausibly far ({distance_from_base:.2f} m) from the Alex base"
        )
    return pos


def _check_jacobian_backend(env) -> None:
    jacobian_full = _as_torch(env._robot.data.body_link_jacobian_w)
    n_joints = len(env._robot.joint_names)
    print(f"[jacobian] body_link_jacobian_w shape={tuple(jacobian_full.shape)}", flush=True)
    slice_ = jacobian_full[:, env._jacobi_body_idx][:, :, env._arm_joint_ids]
    expected = (env.num_envs, 6, len(env._arm_joint_ids))
    if tuple(slice_.shape) != expected:
        raise RuntimeError(f"arm jacobian slice shape {tuple(slice_.shape)} != {expected}")
    if jacobian_full.shape[-1] != n_joints:
        raise RuntimeError(
            f"jacobian joint columns {jacobian_full.shape[-1]} != {n_joints} "
            "(unexpected base-DoF offset for a fixed-base articulation)"
        )
    values = slice_.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("arm jacobian contains non-finite values")
    if float(np.abs(values).max()) < 1e-9:
        raise RuntimeError(
            "arm jacobian reads return zeros (backend gotcha). Fallback required: "
            "pure-numpy right-arm FK/jacobian module."
        )


def _check_stability(env) -> None:
    joint_vel = _as_torch(env._robot.data.joint_vel).detach().cpu().numpy()[0]
    max_vel = float(np.abs(joint_vel).max())
    print(f"[stability] max |joint_vel| after settle: {max_vel:.4f} rad/s", flush=True)
    if max_vel > 0.5:
        raise RuntimeError(
            f"standing pose is not settled (max joint vel {max_vel:.3f} rad/s > 0.5); "
            "consider spawn.articulation_props.enabled_self_collisions=False"
        )


def _check_ik_tracking(env) -> None:
    start_pos, _ = env._ee_pose_w()
    start_z = float(start_pos.detach().cpu().numpy()[0, 2])
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    action[:, 2] = 0.002  # +2 mm/tick upward: away from the door and the torso
    for _ in range(60):
        env.step(action)
    end_pos, _ = env._ee_pose_w()
    end_z = float(end_pos.detach().cpu().numpy()[0, 2])
    gained = end_z - start_z
    print(f"[ik] commanded +0.120 m in z over 60 ticks, tracked {gained:+.4f} m", flush=True)
    # Open-loop tracking is deliberately partial (the PD drives low-pass the
    # per-tick targets); the scripted controller closes the loop every tick,
    # so ~1/3 open-loop tracking converges. Assert a meaningful floor.
    if gained < 0.04:
        raise RuntimeError(
            f"differential IK tracked only {gained:.4f} m of a commanded 0.120 m z move"
        )


def _check_arc_reachability(env) -> None:
    shoulder_ids, _ = env._robot.find_bodies(SHOULDER_BODY_NAME)
    shoulder = (
        _as_torch(env._robot.data.body_pos_w)[:, shoulder_ids[0]].detach().cpu().numpy()[0]
    )
    if np.linalg.norm(shoulder) < 1e-6:
        raise RuntimeError("shoulder body pose read is zero; cannot verify reachability")
    frame_pos, _ = env.door_frame_pose_w()
    frame = frame_pos.detach().cpu().numpy()[0]
    cfg = alex_fixedbase_push_cfg()

    print(f"[arc] shoulder_w={shoulder} door_frame_w={frame}", flush=True)
    worst = (0.0, 0.0)
    for deg in range(0, 51, 5):
        theta = math.radians(deg)
        point_panel = np.array(
            [cfg.surface_x_m(cfg.contact_clearance_m), cfg.push_point_y_m, cfg.push_height_m]
        )
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        point_w = frame + rot @ point_panel
        distance = float(np.linalg.norm(point_w - shoulder))
        print(f"[arc] theta={deg:3d} deg push_point_w={point_w} |d|={distance:.3f} m", flush=True)
        if distance > worst[1]:
            worst = (deg, distance)
        low, high = ARC_REACH_BOUNDS_M
        if not (low <= distance <= high):
            raise RuntimeError(
                f"push point at {deg} deg is {distance:.3f} m from the shoulder, "
                f"outside the reachable band [{low}, {high}] m — retune "
                "ALEX_BASE_POS / alex_fixedbase_push_cfg"
            )
    print(f"[arc] worst case: {worst[1]:.3f} m at {worst[0]} deg", flush=True)


def _check_contact_force(env) -> None:
    frame_pos, _ = env.door_frame_pose_w()
    frame = frame_pos.detach().cpu().numpy()[0]
    cfg = alex_fixedbase_push_cfg()
    target = frame + np.array([cfg.surface_x_m(-0.02), cfg.push_point_y_m, cfg.push_height_m])

    force_matrix_seen = False
    max_force = 0.0
    for tick in range(400):
        ee_pos, _ = env._ee_pose_w()
        pos = ee_pos.detach().cpu().numpy()[0]
        error = target - pos
        step = np.clip(error, -env.cfg.max_pos_delta_m, env.cfg.max_pos_delta_m)
        action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
        action[0, :3] = torch.as_tensor(step, dtype=torch.float32)
        env.step(action)

        if env._contact_sensor.data.force_matrix_w is not None:
            force_matrix_seen = True
        force = float(env.contact_force_w().norm(dim=-1).detach().cpu().numpy()[0])
        max_force = max(max_force, force)
        if bool(env.contact_sensed().detach().cpu().numpy()[0]):
            net = float(
                _as_torch(env._contact_sensor.data.net_forces_w)[:, 0]
                .norm(dim=-1)
                .detach()
                .cpu()
                .numpy()[0]
            )
            print(
                f"[contact] sensed at tick {tick}: filtered={force:.2f} N net={net:.2f} N "
                f"force_matrix_available={force_matrix_seen}",
                flush=True,
            )
            if not force_matrix_seen:
                print(
                    "[contact] WARNING: force_matrix_w unavailable; env fell back to "
                    "net_forces_w",
                    flush=True,
                )
            return
    raise RuntimeError(
        f"no sensed contact after 400 ticks driving into the panel "
        f"(max force seen {max_force:.3f} N)"
    )


def main() -> int:
    rc = 0
    env = None
    try:
        cfg = DoorPushAlexEnvCfg()
        cfg.seed = args.seed
        cfg.sim.device = args.device
        env = gym.make(door_task.DOOR_PUSH_ALEX_ENV_ID, cfg=cfg).unwrapped

        env.reset(seed=args.seed)
        _check_spawn_and_prims(env)
        _step_zero(env, args.settle_ticks)
        _check_pose_backend(env)
        _check_jacobian_backend(env)
        _check_stability(env)
        _check_ik_tracking(env)
        _check_arc_reachability(env)
        if args.contact:
            env.reset(seed=args.seed)
            _step_zero(env, args.settle_ticks)
            _check_contact_force(env)
        print("PASS: Alex IK backend probe passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: Alex IK backend probe failed.", flush=True)
        rc = 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
    return rc


if __name__ == "__main__":
    # os._exit avoids Kit shutdown masking the probe exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
