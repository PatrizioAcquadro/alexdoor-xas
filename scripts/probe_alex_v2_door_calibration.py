#!/usr/bin/env python
"""Candidate-only Alex V2 calibration probes; never writes production calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

from alexdoor_xas.assets.alex_v2_candidate_pd import (  # noqa: E402
    PRODUCTION_RIGHT_ARM_PD_PROFILE,
    apply_right_arm_pd_profile_selection,
    candidate_right_arm_pd_profile_names,
)

POSE_BASE_POSITION_M = (-0.45, -0.38, 0.93)
POSE_BASE_ORIENTATION_XYZW = (0.0, 0.0, 1.0, 0.0)
POSE_READY_JOINT_POSITION_RAD = {
    "RIGHT_SHOULDER_Y": 0.3,
    "RIGHT_SHOULDER_X": 0.0,
    "RIGHT_SHOULDER_Z": 0.0,
    "RIGHT_ELBOW_Y": -0.8,
    "RIGHT_WRIST_Z": 0.0,
    "RIGHT_WRIST_X": 0.0,
}
POSE_CONTACT_NORMAL_WORLD = (-1.0, 0.0, 0.0)
RIGHT_GRIPPER_BODY_NAME = "RIGHT_GRIPPER_Z_LINK"
RIGHT_SHOULDER_BODY_NAME = "RIGHT_SHOULDER_Z_LINK"
RESET_VELOCITY_BOUND_RAD_S = 0.5
RESET_GRACE_STEPS = 30
RESET_MEASURED_STEPS = 90


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite float greater than zero")
    return parsed


parser = argparse.ArgumentParser(description="Alex V2 door calibration probe")
parser.add_argument("--stage", choices=("spawn", "pose"), default="spawn")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument(
    "--damping-scale",
    type=_positive_float,
    default=1.0,
    help=(
        "Additional candidate-only multiplier applied after the production "
        "fixed-base V2 damping scale."
    ),
)
parser.add_argument(
    "--right-arm-pd-profile",
    choices=(
        "none",
        PRODUCTION_RIGHT_ARM_PD_PROFILE,
        *candidate_right_arm_pd_profile_names(),
    ),
    default="none",
    help=(
        "Candidate-only exact right-arm PD override. "
        f"'{PRODUCTION_RIGHT_ARM_PD_PROFILE}' is the canonical production IK40 spelling; "
        "The default 'none' also preserves production exactly."
    ),
)
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Candidate-only: disable self-collision in both URDF import and articulation settings.",
)
parser.add_argument(
    "--door-normal-link",
    nargs=3,
    type=float,
    default=(1.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help=(
        "Spawn-stage candidate contact normal in RIGHT_GRIPPER_Z_LINK coordinates; "
        "pose derives its link normal from the fixed world contact direction."
    ),
)
parser.add_argument(
    "--out",
    type=Path,
    default=None,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out is None:
    args.out = Path(f"outputs/door_push_alex_v2/calibration/v0/{args.stage}_probe.json")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from alexdoor_xas.assets.alex_v2 import (  # noqa: E402
    build_alex_v2_door_asset,
    load_alex_v2_articulation_cfg,
)
from alexdoor_xas.assets.alex_v2_contract import EXPECTED_RUNTIME_JOINTS  # noqa: E402
from alexdoor_xas.assets.alex_v2_tool_frame import (  # noqa: E402
    derive_right_gripper_tool_frame,
)
from alexdoor_xas.calibration.alex_v2_door_authoring import (  # noqa: E402
    make_reset_stability_evidence,
)
from alexdoor_xas.kinematics.offset_point import (  # noqa: E402
    compose_offset_pose_xyzw,
    link_jacobian_to_point,
    world_vector_to_link_xyzw,
)


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _version(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _apply_candidate_settings(cfg) -> dict:
    if args.right_arm_pd_profile != "none" and args.damping_scale != 1.0:
        raise ValueError(
            "--right-arm-pd-profile cannot be combined with --damping-scale; "
            "each named profile already contains exact final right-arm gains"
        )
    if args.right_arm_pd_profile != "none" and args.disable_self_collision:
        raise ValueError(
            "named right-arm PD candidates require self-collision to remain enabled"
        )
    self_collision_enabled = not args.disable_self_collision
    cfg.spawn.self_collision = self_collision_enabled
    cfg.spawn.articulation_props.enabled_self_collisions = self_collision_enabled

    scaled_damping = {}
    for actuator_name, actuator_cfg in cfg.actuators.items():
        if not isinstance(actuator_cfg.damping, dict):
            raise TypeError(
                f"actuator {actuator_name!r} damping must be a mapping for candidate scaling"
            )
        actuator_cfg.damping = {
            joint_expression: float(value) * args.damping_scale
            for joint_expression, value in actuator_cfg.damping.items()
        }
        scaled_damping[actuator_name] = dict(actuator_cfg.damping)

    selection = apply_right_arm_pd_profile_selection(
        cfg, profile_name=args.right_arm_pd_profile
    )

    return {
        "scope": "calibration_probe_only",
        "damping_scale": args.damping_scale,
        "uses_production_damping_default": args.damping_scale == 1.0,
        "right_arm_pd_profile": args.right_arm_pd_profile,
        "effective_right_arm_pd_profile": selection["effective_profile"],
        "uses_production_right_arm_pd": selection["uses_production_right_arm_pd"],
        "disable_self_collision": args.disable_self_collision,
        "self_collision_fields": {
            "cfg.spawn.self_collision": bool(cfg.spawn.self_collision),
            "cfg.spawn.articulation_props.enabled_self_collisions": bool(
                cfg.spawn.articulation_props.enabled_self_collisions
            ),
        },
        "actuator_damping": scaled_damping,
        "right_arm_pd": selection["candidate_override"],
    }


def _apply_pose_initial_state(cfg) -> dict:
    cfg.init_state.pos = POSE_BASE_POSITION_M
    cfg.init_state.rot = POSE_BASE_ORIENTATION_XYZW
    explicitly_posed = tuple(POSE_READY_JOINT_POSITION_RAD)
    exclude = "|".join(explicitly_posed)
    cfg.init_state.joint_pos = {
        **POSE_READY_JOINT_POSITION_RAD,
        f"(?!(?:{exclude})$).*": 0.0,
    }
    return {
        "base_position_world_m": list(POSE_BASE_POSITION_M),
        "base_orientation_world_xyzw": list(POSE_BASE_ORIENTATION_XYZW),
        "joint_position_rad": dict(POSE_READY_JOINT_POSITION_RAD),
    }


def main() -> int:
    grace_steps = RESET_GRACE_STEPS if args.stage == "pose" else 0
    evidence = {
        "stage": args.stage,
        "status": "started",
        "candidate_only": True,
        "candidate_settings": {
            "scope": "calibration_probe_only",
            "damping_scale": args.damping_scale,
            "uses_production_damping_default": args.damping_scale == 1.0,
            "disable_self_collision": args.disable_self_collision,
            "right_arm_pd_profile": args.right_arm_pd_profile,
        },
        "production_calibration_written": False,
        "steps": args.steps,
        "reset_grace_steps": grace_steps,
        "reset_measured_steps": args.steps - grace_steps,
        "joint_velocity_peak_trace": [],
        "max_joint_velocity_rad_s": None,
        "peak_joint_name": None,
        "peak_joint_step": None,
        "peak_environment_index": None,
    }
    simulation = None
    exit_code = 0
    try:
        minimum_steps = grace_steps + (RESET_MEASURED_STEPS if grace_steps else 1)
        if args.steps < minimum_steps:
            raise ValueError(
                f"--steps must be at least {minimum_steps} for stage {args.stage!r}"
            )
        asset, asset_ref = build_alex_v2_door_asset()
        cfg = load_alex_v2_articulation_cfg(fix_base=True).replace(
            prim_path="/World/AlexV2CalibrationProbe"
        )
        if args.stage == "pose":
            evidence["candidate_pose_initial_state"] = _apply_pose_initial_state(cfg)
        evidence["candidate_settings"] = _apply_candidate_settings(cfg)
        simulation = SimulationContext(SimulationCfg(device=args.device, dt=1.0 / 120.0))
        robot = Articulation(cfg)
        simulation.reset()
        robot.reset()
        if args.stage == "pose":
            robot.set_joint_position_target_index(
                target=_tensor(robot.data.default_joint_pos)
            )
        actual_joints = tuple(robot.joint_names)
        if actual_joints != EXPECTED_RUNTIME_JOINTS:
            raise RuntimeError("runtime joint order differs from the 29-name V2 manifest contract")

        grace_peak_velocity = 0.0
        grace_peak_joint_name = None
        grace_peak_joint_step = None
        grace_peak_environment_index = None
        measured_peak_velocity = -1.0
        measured_peak_joint_name = None
        measured_peak_joint_step = None
        measured_peak_environment_index = None
        peak_velocity = -1.0
        peak_joint_name = None
        peak_joint_step = None
        peak_environment_index = None
        for step_index in range(args.steps):
            step_number = step_index + 1
            robot.write_data_to_sim()
            simulation.step(render=False)
            robot.update(simulation.get_physics_dt())
            step_position = _tensor(robot.data.joint_pos)
            finite_position = torch.isfinite(step_position)
            if not finite_position.all():
                nonfinite_flat_index = int(
                    (~finite_position).reshape(-1).nonzero(as_tuple=False)[0, 0].item()
                )
                nonfinite_joint_name = actual_joints[nonfinite_flat_index % len(actual_joints)]
                evidence.update(
                    {
                        "failed_step": step_number,
                        "failed_joint_name": nonfinite_joint_name,
                    }
                )
                raise RuntimeError(
                    "Alex V2 spawn produced non-finite joint position "
                    f"at step {step_number} on joint {nonfinite_joint_name}"
                )
            step_velocity = _tensor(robot.data.joint_vel)
            finite_velocity = torch.isfinite(step_velocity)
            if not finite_velocity.all():
                nonfinite_flat_index = int(
                    (~finite_velocity).reshape(-1).nonzero(as_tuple=False)[0, 0].item()
                )
                nonfinite_joint_name = actual_joints[nonfinite_flat_index % len(actual_joints)]
                evidence.update(
                    {
                        "failed_step": step_number,
                        "failed_joint_name": nonfinite_joint_name,
                    }
                )
                raise RuntimeError(
                    "Alex V2 spawn produced non-finite joint velocity "
                    f"at step {step_number} on joint {nonfinite_joint_name}"
                )

            absolute_velocity = step_velocity.abs().reshape(-1)
            step_flat_index = int(absolute_velocity.argmax().item())
            step_peak_velocity = float(absolute_velocity[step_flat_index].item())
            step_joint_name = actual_joints[step_flat_index % len(actual_joints)]
            step_environment_index = step_flat_index // len(actual_joints)
            evidence["joint_velocity_peak_trace"].append(
                {
                    "step": step_number,
                    "peak_velocity_rad_s": step_peak_velocity,
                    "peak_joint_name": step_joint_name,
                    "environment_index": step_environment_index,
                }
            )
            if step_peak_velocity > peak_velocity:
                peak_velocity = step_peak_velocity
                peak_joint_name = step_joint_name
                peak_joint_step = step_number
                peak_environment_index = step_environment_index
                evidence.update(
                    {
                        "max_joint_velocity_rad_s": peak_velocity,
                        "peak_joint_name": peak_joint_name,
                        "peak_joint_step": peak_joint_step,
                        "peak_environment_index": peak_environment_index,
                    }
                )
            if step_number <= grace_steps and step_peak_velocity > grace_peak_velocity:
                grace_peak_velocity = step_peak_velocity
                grace_peak_joint_name = step_joint_name
                grace_peak_joint_step = step_number
                grace_peak_environment_index = step_environment_index
            elif (
                step_number > grace_steps
                and step_peak_velocity > measured_peak_velocity
            ):
                measured_peak_velocity = step_peak_velocity
                measured_peak_joint_name = step_joint_name
                measured_peak_joint_step = step_number
                measured_peak_environment_index = step_environment_index

        joint_pos = _tensor(robot.data.joint_pos)
        joint_vel = _tensor(robot.data.joint_vel)
        if not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all():
            raise RuntimeError("Alex V2 spawn produced non-finite joint state")
        reset_stability = make_reset_stability_evidence(
            finite_state=True,
            grace_peak_abs_joint_velocity_rad_s=grace_peak_velocity,
            measured_peak_abs_joint_velocity_rad_s=measured_peak_velocity,
            grace_steps=grace_steps,
            measured_steps=args.steps - grace_steps,
            bound_rad_s=RESET_VELOCITY_BOUND_RAD_S,
        )
        reset_stability.update(
            {
                "grace_peak_joint_name": grace_peak_joint_name,
                "grace_peak_joint_step": grace_peak_joint_step,
                "grace_peak_environment_index": grace_peak_environment_index,
                "measured_peak_joint_name": measured_peak_joint_name,
                "measured_peak_joint_step": measured_peak_joint_step,
                "measured_peak_environment_index": measured_peak_environment_index,
                "full_window_peak_abs_joint_velocity_rad_s": peak_velocity,
                "full_window_peak_joint_name": peak_joint_name,
                "full_window_peak_joint_step": peak_joint_step,
                "full_window_peak_environment_index": peak_environment_index,
            }
        )
        evidence["reset_stability"] = reset_stability
        if not reset_stability["passed"]:
            raise RuntimeError(
                "Alex V2 reset is unstable after the grace window: peak joint velocity "
                f"{measured_peak_velocity:.6f} rad/s at step {measured_peak_joint_step} "
                f"on joint {measured_peak_joint_name}"
            )
        body_ids, _ = robot.find_bodies(RIGHT_GRIPPER_BODY_NAME)
        if len(body_ids) != 1:
            raise RuntimeError(
                f"{RIGHT_GRIPPER_BODY_NAME} did not resolve to exactly one body"
            )
        body_id = body_ids[0]
        body_idx = body_id - 1
        link_positions_w = _tensor(robot.data.body_link_pos_w)[:, body_id]
        link_orientations_w_xyzw = _tensor(robot.data.body_link_quat_w)[:, body_id]
        if not torch.isfinite(link_positions_w).all() or not torch.isfinite(
            link_orientations_w_xyzw
        ).all():
            raise RuntimeError("Alex V2 gripper link pose is non-finite")
        jacobians = _tensor(robot.data.body_link_jacobian_w)
        jacobian = jacobians[:, body_idx]
        if not torch.isfinite(jacobian).all() or float(jacobian.abs().max()) <= 1e-9:
            raise RuntimeError("Alex V2 gripper Jacobian is non-finite or identically zero")

        shared_diagnostics = {
            "gripper_link_pose_world": {
                "position_m": link_positions_w[0].detach().cpu().tolist(),
                "orientation_xyzw": link_orientations_w_xyzw[0]
                .detach()
                .cpu()
                .tolist(),
            },
            "link_jacobian": {
                "shape": list(jacobian.shape),
                "finite": True,
                "max_abs": float(jacobian.abs().max().item()),
            },
        }
        pose_diagnostics = None
        if args.stage == "pose":
            desired_normal_w = torch.tensor(
                POSE_CONTACT_NORMAL_WORLD,
                dtype=link_orientations_w_xyzw.dtype,
                device=link_orientations_w_xyzw.device,
            )
            desired_normal_link = world_vector_to_link_xyzw(
                link_orientations_w_xyzw, desired_normal_w
            )
            tool = derive_right_gripper_tool_frame(
                asset.manifest,
                desired_normal_link[0].detach().cpu().tolist(),
            )
            tool_translation_link = torch.tensor(
                tool.translation_m,
                dtype=link_positions_w.dtype,
                device=link_positions_w.device,
            )
            tool_orientation_link_xyzw = torch.tensor(
                tool.orientation_xyzw,
                dtype=link_orientations_w_xyzw.dtype,
                device=link_orientations_w_xyzw.device,
            )
            tool_positions_w, tool_orientations_w_xyzw = compose_offset_pose_xyzw(
                link_positions_w,
                link_orientations_w_xyzw,
                tool_translation_link,
                tool_orientation_link_xyzw,
            )
            point_jacobian = link_jacobian_to_point(
                jacobian,
                link_orientations_w_xyzw,
                tool_translation_link,
            )
            point_jacobian_finite = bool(torch.isfinite(point_jacobian).all())
            if not point_jacobian_finite or float(point_jacobian.abs().max()) <= 1e-9:
                raise RuntimeError(
                    "Alex V2 tool-point Jacobian is non-finite or identically zero"
                )
            shoulder_ids, _ = robot.find_bodies(RIGHT_SHOULDER_BODY_NAME)
            if len(shoulder_ids) != 1:
                raise RuntimeError(
                    f"{RIGHT_SHOULDER_BODY_NAME} did not resolve to exactly one body"
                )
            shoulder_position_w = _tensor(robot.data.body_link_pos_w)[
                :, shoulder_ids[0]
            ]
            if not torch.isfinite(shoulder_position_w).all():
                raise RuntimeError("Alex V2 shoulder position is non-finite")
            shoulder_to_tool_reach = torch.linalg.vector_norm(
                tool_positions_w - shoulder_position_w, dim=-1
            )
            if not torch.isfinite(shoulder_to_tool_reach).all():
                raise RuntimeError("Alex V2 shoulder-to-tool reach is non-finite")
            pose_diagnostics = {
                "desired_contact_normal_world": list(POSE_CONTACT_NORMAL_WORLD),
                "desired_contact_normal_link": desired_normal_link[0]
                .detach()
                .cpu()
                .tolist(),
                "candidate_tool_world_pose": {
                    "position_m": tool_positions_w[0].detach().cpu().tolist(),
                    "orientation_xyzw": tool_orientations_w_xyzw[0]
                    .detach()
                    .cpu()
                    .tolist(),
                },
                "shoulder_position_world_m": shoulder_position_w[0]
                .detach()
                .cpu()
                .tolist(),
                "shoulder_to_tool_reach_m": float(
                    shoulder_to_tool_reach[0].item()
                ),
                "point_jacobian": {
                    "shape": list(point_jacobian.shape),
                    "finite": point_jacobian_finite,
                    "max_abs": float(point_jacobian.abs().max().item()),
                },
            }
        else:
            tool = derive_right_gripper_tool_frame(
                asset.manifest, args.door_normal_link
            )
        evidence.update(
            {
                "status": "passed",
                "robot_asset": asset_ref.to_dict(),
                "urdf_path": str(asset.urdf_path),
                "runtime_joint_order": list(actual_joints),
                "max_joint_velocity_rad_s": peak_velocity,
                "peak_joint_name": peak_joint_name,
                "peak_joint_step": peak_joint_step,
                "peak_environment_index": peak_environment_index,
                "jacobian_shape": list(jacobian.shape),
                "candidate_tool_frame": tool.to_dict(),
                "shared_diagnostics": shared_diagnostics,
                "runtime_versions": {
                    "isaac_sim": _version(Path.home() / "isaacsim" / "VERSION"),
                    "isaac_lab": _version(Path.home() / "IsaacLab" / "VERSION"),
                },
                "next_required_stages": [
                    "measured_door_normal",
                    "reach_shell",
                    "contact_behavior",
                    "fixed_scripted_baseline",
                    "randomized_scripted_baseline",
                ],
            }
        )
        if pose_diagnostics is not None:
            evidence["pose_diagnostics"] = pose_diagnostics
    except Exception as error:  # noqa: BLE001 - evidence captures the live blocker.
        evidence.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 2
    finally:
        if simulation is not None:
            try:
                simulation.clear_instance()
            except Exception as error:  # noqa: BLE001 - retain shutdown evidence.
                cleanup_traceback = traceback.format_exc()
                evidence.update(
                    {
                        "cleanup_error_type": type(error).__name__,
                        "cleanup_error": str(error),
                        "cleanup_traceback": cleanup_traceback,
                    }
                )
                if exit_code == 0:
                    evidence.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": f"simulation cleanup failed: {error}",
                            "traceback": cleanup_traceback,
                        }
                    )
                    exit_code = 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": evidence["status"], "out": str(args.out)}))
    return exit_code


def _close_simulation_app(exit_code: int) -> None:
    """Close Kit while preserving failures across Isaac Sim API versions."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        simulation_app.close(exit_code=exit_code)
    except TypeError as error:
        if "exit_code" not in str(error):
            raise
        if exit_code != 0:
            os._exit(exit_code)
        simulation_app.close()


if __name__ == "__main__":
    result = main()
    _close_simulation_app(result)
    sys.exit(result)
