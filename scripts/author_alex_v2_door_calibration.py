#!/usr/bin/env python
"""Author the Alex V2 door calibration after its internal safety checks pass.

This is a mutating maintenance command, not a supported verifier: on success it
writes the production calibration. Do not run it as part of routine validation.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/author_alex_v2_door_calibration.py --viz none --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

EVIDENCE_DIR = Path("outputs/door_push_alex_v2/calibration/v0")
EVIDENCE_PATH = EVIDENCE_DIR / "author_alex_v2_door_calibration.evidence.json"
CANDIDATE_PATH = EVIDENCE_DIR / "alex_v2_door_calibration.candidate.json"

parser = argparse.ArgumentParser(description="Author the Alex V2 door calibration")
parser.add_argument("--fixed-seed", type=int, default=0)
parser.add_argument("--max-ticks", type=int, default=600)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Close SimulationApp before exiting; useful for diagnosing Kit shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


evidence: dict[str, Any] = {
    "schema_version": "alexdoor.author_alex_v2_door_calibration.evidence.v1",
    "status": "starting",
    "production_calibration_written": False,
    "candidate_path": str(CANDIDATE_PATH),
    "production_path": "configs/alex_v2_door.json",
    "gates": {},
    "gate_evidence": {},
}
_write_json(EVIDENCE_PATH, evidence)


def _record_unhandled_exception(exc_type: type[BaseException], error: BaseException, tb) -> None:
    evidence.update(
        {
            "status": "failed",
            "error": f"{exc_type.__name__}: {error}",
            "traceback": "".join(traceback.format_exception(exc_type, error, tb))[-4000:],
        }
    )
    _write_json(EVIDENCE_PATH, evidence)
    sys.__excepthook__(exc_type, error, tb)


sys.excepthook = _record_unhandled_exception

try:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
except Exception:  # noqa: BLE001
    evidence.update(
        {
            "status": "failed",
            "error": "AppLauncher construction failed",
            "traceback": traceback.format_exc()[-4000:],
        }
    )
    _write_json(EVIDENCE_PATH, evidence)
    raise

# Runtime imports must follow AppLauncher construction.
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.assets.alex_v2 import build_alex_v2_door_asset  # noqa: E402
from alexdoor_xas.assets.alex_v2_contract import EXPECTED_RUNTIME_JOINTS  # noqa: E402
from alexdoor_xas.assets.alex_v2_tool_frame import (  # noqa: E402
    derive_right_gripper_tool_frame,
)
from alexdoor_xas.calibration.alex_v2_door import (  # noqa: E402
    default_calibration_path,
    load_alex_v2_door_calibration,
)
from alexdoor_xas.calibration.alex_v2_door_authoring import (  # noqa: E402
    CONTROLLER,
    REQUIRED_GATES,
    all_required_gates_pass,
    compose_calibration_payload,
    distance_envelope,
    envelope_within_shell,
    make_reset_stability_evidence,
    write_calibration,
)
from alexdoor_xas.data_engine import (  # noqa: E402
    DataEngineCfg,
    EpisodePlanItem,
    plan_episodes,
    run_episode,
)
from alexdoor_xas.envs.door_task.alex_v2_runtime import (  # noqa: E402
    ALEX_V2_LIMITATIONS,
    require_current_collision_tool_frame,
)
from alexdoor_xas.envs.door_task.door_push_alex_v2_calibration_env import (  # noqa: E402
    DoorPushAlexV2CalibrationEnv,
)
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.eval.sanity import (  # noqa: E402
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from alexdoor_xas.policies.scripted.door_push_alex_v2 import (  # noqa: E402
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)

RESET_VELOCITY_BOUND_RAD_S = 0.5
RESET_GRACE_STEPS = 30
RESET_MEASURED_STEPS = 90
JACOBIAN_NONZERO_TOL = 1.0e-9
RANDOMIZED_SEEDS = tuple(range(5))


def _tensor(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _numpy(value: Any) -> np.ndarray:
    tensor = _tensor(value)
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _sample_reach_distance(env: DoorPushAlexV2CalibrationEnv) -> float:
    shoulder = _numpy(env.shoulder_position_world_m())[0]
    tool = _numpy(env.ee_pose_w()[0])[0]
    return float(np.linalg.norm(tool - shoulder))


def _settle_reset(env: DoorPushAlexV2CalibrationEnv) -> dict[str, Any]:
    env.reset(seed=args.fixed_seed)
    zero_action = torch.zeros(
        (env.num_envs, env.cfg.action_space), dtype=torch.float32, device=env.device
    )

    def sample_state() -> tuple[bool, float]:
        state = env.robot_joint_state()
        finite = all(np.isfinite(np.asarray(value)).all() for value in state.values())
        absolute_velocity = np.abs(np.asarray(state["joint_vel"]))
        finite_velocity = absolute_velocity[np.isfinite(absolute_velocity)]
        peak = float(finite_velocity.max()) if finite_velocity.size else 0.0
        return finite, peak

    finite, grace_peak_velocity = sample_state()
    for _ in range(RESET_GRACE_STEPS):
        env.step(zero_action)
        step_finite, step_peak = sample_state()
        finite = finite and step_finite
        grace_peak_velocity = max(grace_peak_velocity, step_peak)

    measured_peak_velocity = 0.0
    for _ in range(RESET_MEASURED_STEPS):
        env.step(zero_action)
        step_finite, step_peak = sample_state()
        finite = finite and step_finite
        measured_peak_velocity = max(measured_peak_velocity, step_peak)

    return make_reset_stability_evidence(
        finite_state=finite,
        grace_peak_abs_joint_velocity_rad_s=grace_peak_velocity,
        measured_peak_abs_joint_velocity_rad_s=measured_peak_velocity,
        grace_steps=RESET_GRACE_STEPS,
        measured_steps=RESET_MEASURED_STEPS,
        bound_rad_s=RESET_VELOCITY_BOUND_RAD_S,
    )


def _jacobian_evidence(env: DoorPushAlexV2CalibrationEnv) -> dict[str, Any]:
    link = _tensor(env._robot.data.body_link_jacobian_w)[:, env._jacobi_body_idx][  # noqa: SLF001
        :, :, env._arm_joint_ids  # noqa: SLF001
    ]
    tool = env.point_jacobian_w()
    link_finite = bool(torch.isfinite(link).all())
    tool_finite = bool(torch.isfinite(tool).all())
    link_max = float(link.abs().max().item())
    tool_max = float(tool.abs().max().item())
    return {
        "link_shape": list(link.shape),
        "tool_shape": list(tool.shape),
        "link_finite": link_finite,
        "tool_finite": tool_finite,
        "link_max_abs": link_max,
        "tool_max_abs": tool_max,
        "passed": bool(
            link_finite
            and tool_finite
            and link_max > JACOBIAN_NONZERO_TOL
            and tool_max > JACOBIAN_NONZERO_TOL
        ),
    }


def _episode_evidence(episode: Any, distances: list[float]) -> tuple[dict[str, Any], bool]:
    sanity = check_alex_episode(episode, force_error_n=FORCE_DATASET_LIMIT_N)
    outcome = episode.outcome
    runtime_notes = "" if outcome is None else str(outcome.notes)
    completed = outcome is not None and not runtime_notes and episode.n_steps <= args.max_ticks
    envelope = distance_envelope(distances)
    shell_ok = envelope_within_shell(distances, (0.2, 0.8))
    summary = {
        "seed": episode.meta.seed,
        "n_steps": episode.n_steps,
        "completed_within_tick_budget": completed,
        "success": bool(outcome is not None and outcome.success),
        "final_door_angle_rad": None if outcome is None else outcome.final_door_angle,
        "failure_label": None if outcome is None else outcome.failure_label,
        "runtime_notes": runtime_notes,
        "controller_done": bool(episode.extras.get("controller_done")),
        "controller_timed_out": bool(episode.extras.get("controller_timed_out")),
        "sanity_errors": list(sanity.errors),
        "sanity_warnings": list(sanity.warnings),
        "reach_envelope_m": None if envelope is None else list(envelope),
        "reach_shell_passed": shell_ok,
        "force_diagnostics": contact_force_diagnostics(
            episode, force_limit_n=FORCE_DATASET_LIMIT_N
        ),
    }
    return summary, bool(completed and sanity.ok and shell_ok)


def _run_recorded_episode(
    env: DoorPushAlexV2CalibrationEnv,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg,
    controller_cfg: Any,
) -> tuple[Any, list[float]]:
    distances: list[float] = []

    def sample(_tick: int) -> None:
        distances.append(_sample_reach_distance(env))

    episode = run_episode(
        env,
        item,
        engine_cfg,
        controller_cfg=controller_cfg,
        render_hook=sample,
    )
    return episode, distances


def _fixed_gates(
    env: DoorPushAlexV2CalibrationEnv,
    engine_cfg: DataEngineCfg,
    controller_cfg: Any,
) -> tuple[dict[str, Any], bool, bool]:
    episode, distances = _run_recorded_episode(
        env,
        EpisodePlanItem(seed=args.fixed_seed),
        engine_cfg,
        controller_cfg,
    )
    summary, baseline_common = _episode_evidence(episode, distances)
    forces = np.asarray(
        [float(step.contact.get("force_n", 0.0)) for step in episode.steps],
        dtype=np.float64,
    )
    push_forces = np.asarray(
        [
            float(step.contact.get("force_n", 0.0))
            for step in episode.steps
            if step.safety.get("controller_phase") == "push"
        ],
        dtype=np.float64,
    )
    all_forces_finite = bool(np.isfinite(forces).all())
    peak_push_force = float(push_forces.max()) if push_forces.size else 0.0
    contact_passed = bool(
        all_forces_finite
        and np.isfinite(push_forces).all()
        and peak_push_force >= CONTROLLER["contact_force_threshold_n"]
        and summary["force_diagnostics"]["force_admission_passed"]
    )
    outcome = episode.outcome
    success_angle_passed = bool(outcome is not None and outcome.final_door_angle >= math.pi / 4.0)
    summary.update(
        {
            "success_angle_rad": math.pi / 4.0,
            "success_angle_passed": success_angle_passed,
            "all_contact_forces_finite": all_forces_finite,
            "push_phase_samples": int(push_forces.size),
            "peak_push_contact_force_n": peak_push_force,
            "contact_force_threshold_n": CONTROLLER["contact_force_threshold_n"],
        }
    )
    return summary, bool(baseline_common and success_angle_passed), contact_passed


def _randomized_gate(
    env: DoorPushAlexV2CalibrationEnv,
    engine_cfg: DataEngineCfg,
    controller_cfg: Any,
    variation_bounds: Any,
) -> tuple[dict[str, Any], bool]:
    summaries: list[dict[str, Any]] = []
    all_common = True
    successes = 0
    all_distances: list[float] = []
    for seed in RANDOMIZED_SEEDS:
        item = plan_episodes(0, 1, seed, bounds=variation_bounds)[0]
        episode, distances = _run_recorded_episode(env, item, engine_cfg, controller_cfg)
        summary, common = _episode_evidence(episode, distances)
        summaries.append(summary)
        all_common = all_common and common
        successes += int(summary["success"])
        all_distances.extend(distances)
    envelope = distance_envelope(all_distances)
    passed = bool(all_common and successes >= 4)
    return (
        {
            "seeds": list(RANDOMIZED_SEEDS),
            "successes": successes,
            "required_successes": 4,
            "all_sanity_and_shell_checks_passed": all_common,
            "reach_envelope_m": None if envelope is None else list(envelope),
            "episodes": summaries,
        },
        passed,
    )


def main() -> int:
    env = None
    gates = {name: False for name in REQUIRED_GATES}
    evidence["gates"] = gates
    evidence["status"] = "running"
    _write_json(EVIDENCE_PATH, evidence)
    try:
        if args.max_ticks < 1:
            raise ValueError("--max-ticks must be at least 1")

        asset, robot_asset = build_alex_v2_door_asset()
        runtime_versions = {
            "isaac_sim": (Path.home() / "isaacsim" / "VERSION").read_text().strip(),
            "isaac_lab": (Path.home() / "IsaacLab" / "VERSION").read_text().strip(),
        }
        candidate = compose_calibration_payload(asset.manifest)
        _write_json(CANDIDATE_PATH, candidate)
        calibration = load_alex_v2_door_calibration(CANDIDATE_PATH)
        evidence.update(
            {
                "robot_asset": robot_asset.to_dict(),
                "runtime_versions": runtime_versions,
            }
        )

        cfg = DoorPushAlexV2EnvCfg()
        cfg.seed = args.fixed_seed
        cfg.sim.device = args.device
        env = DoorPushAlexV2CalibrationEnv(
            cfg,
            candidate_calibration_path=CANDIDATE_PATH,
            render_mode=None,
        )

        actual_joint_order = tuple(env.robot_joint_names())
        gates["exact_runtime_joint_order"] = actual_joint_order == EXPECTED_RUNTIME_JOINTS
        evidence["gate_evidence"]["exact_runtime_joint_order"] = {
            "actual": list(actual_joint_order),
            "expected": list(EXPECTED_RUNTIME_JOINTS),
            "passed": gates["exact_runtime_joint_order"],
        }

        reset = _settle_reset(env)
        gates["reset_stability"] = reset["passed"]
        evidence["gate_evidence"]["reset_stability"] = reset

        jacobians = _jacobian_evidence(env)
        gates["finite_jacobians"] = jacobians["passed"]
        evidence["gate_evidence"]["finite_jacobians"] = jacobians

        require_current_collision_tool_frame(asset.manifest, calibration)
        derived_tool_frame_full = derive_right_gripper_tool_frame(
            asset.manifest, candidate["tool_frame"]["contact_normal_link"]
        ).to_dict()
        derived_tool_frame = {
            field: derived_tool_frame_full[field] for field in candidate["tool_frame"]
        }
        tool_match = derived_tool_frame == candidate["tool_frame"]
        gates["collision_tool_frame"] = tool_match
        evidence["gate_evidence"]["collision_tool_frame"] = {
            "derived": derived_tool_frame,
            "candidate": candidate["tool_frame"],
            "passed": tool_match,
        }

        engine_cfg = DataEngineCfg(
            task=paths.ALEX_V2_TASK,
            robot=ALEX_V2_ROBOT_TAG,
            limitations=ALEX_V2_LIMITATIONS,
            max_ticks=args.max_ticks,
        )
        controller_cfg = alex_v2_push_cfg(calibration)
        variation_bounds = alex_v2_variation_bounds(calibration)

        fixed, fixed_passed, contact_passed = _fixed_gates(env, engine_cfg, controller_cfg)
        gates["fixed_scripted_baseline"] = fixed_passed
        gates["contact_behavior"] = contact_passed
        evidence["gate_evidence"]["fixed_scripted_baseline"] = fixed
        evidence["gate_evidence"]["contact_behavior"] = {
            "all_contact_forces_finite": fixed["all_contact_forces_finite"],
            "push_phase_samples": fixed["push_phase_samples"],
            "peak_push_contact_force_n": fixed["peak_push_contact_force_n"],
            "threshold_n": fixed["contact_force_threshold_n"],
            "upper_force_limit_n": FORCE_DATASET_LIMIT_N,
            "upper_force_gate_passed": fixed["force_diagnostics"]["upper_force_gate_passed"],
            "non_negative_force_gate_passed": fixed["force_diagnostics"][
                "non_negative_force_gate_passed"
            ],
            "force_admission_passed": fixed["force_diagnostics"]["force_admission_passed"],
            "force_diagnostics": fixed["force_diagnostics"],
            "raw_contact_capacity": env.cfg.ee_contact.max_contact_data_count_per_prim,
            "filter_prim_paths_expr": list(env.cfg.ee_contact.filter_prim_paths_expr),
            "contact_actor_paths_seen": list(env.contact_actor_paths_seen()),
            "force_semantics": "exact door actor selected from PhysX raw GPU contacts",
            "passed": contact_passed,
        }

        randomized, randomized_passed = _randomized_gate(
            env, engine_cfg, controller_cfg, variation_bounds
        )
        gates["randomized_scripted_baseline"] = randomized_passed
        evidence["gate_evidence"]["randomized_scripted_baseline"] = randomized

        if not all_required_gates_pass(gates):
            failed = [name for name in REQUIRED_GATES if not gates[name]]
            raise RuntimeError("live Alex V2 calibration gates failed: " + ", ".join(failed))

        production_path = default_calibration_path()
        write_calibration(production_path, candidate, gates)
        evidence.update(
            {
                "production_calibration_written": True,
                "production_path": str(production_path),
            }
        )
        load_alex_v2_door_calibration(production_path)
        evidence.update(
            {
                "status": "passed",
            }
        )
        print(f"PASS: wrote Alex V2 calibration to {production_path}", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001
        evidence.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        traceback.print_exc()
        print("FAIL: Alex V2 calibration authoring failed.", flush=True)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001
                evidence.setdefault("cleanup_errors", []).append(
                    f"env.close: {type(error).__name__}: {error}"
                )
        _write_json(EVIDENCE_PATH, evidence)
        print(f"[evidence] {EVIDENCE_PATH}", flush=True)


if __name__ == "__main__":
    result = main()
    if args.clean_shutdown:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            result = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
