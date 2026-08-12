#!/usr/bin/env python
"""Evaluate an ACT or Diffusion best checkpoint under a closed-loop protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from alexdoor_xas import paths
from alexdoor_xas.policies.act.config import act_config_from_dict
from alexdoor_xas.policies.common.closed_loop import (
    evaluation_preflight,
    validate_evaluation_protocol,
)
from alexdoor_xas.policies.common.runs import load_resolved_config
from alexdoor_xas.policies.diffusion.config import diffusion_config_from_dict

POLICIES = ("act", "diffusion")

parser = argparse.ArgumentParser(description="AlexDoor-XAS policy closed-loop evaluation")
parser.add_argument("--checkpoint", help="Training run checkpoints/best.pt.")
parser.add_argument(
    "--protocol",
    help="Complete evaluation protocol JSON; omit to use the source run's frozen protocol.",
)
parser.add_argument(
    "--trace-rollout",
    action="append",
    default=[],
    help="Additionally retain this rollout key in traces/; may be repeated.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting.",
)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if unknown:
    parser.error(
        "evaluation does not accept config overrides; use a complete --protocol JSON: "
        + " ".join(unknown)
    )
if args.checkpoint is None:
    parser.error("the following argument is required: --checkpoint")

checkpoint_path = Path(args.checkpoint).expanduser().resolve()
if not checkpoint_path.is_file():
    parser.error(f"checkpoint not found: {checkpoint_path}")
source_run = checkpoint_path.parent.parent
source_resolved = load_resolved_config(source_run)
policy_name = source_resolved.get("policy")
if source_resolved.get("run_type") != "training" or policy_name not in POLICIES:
    parser.error("--checkpoint must be best.pt from an ACT or Diffusion training run")
try:
    policy_cfg = (
        act_config_from_dict(source_resolved["config"])
        if policy_name == "act"
        else diffusion_config_from_dict(source_resolved["config"])
    )
except (KeyError, ValueError) as error:
    parser.error(str(error))
protocol = (
    json.loads(Path(args.protocol).expanduser().read_text())
    if args.protocol
    else source_resolved["evaluation_protocol"]
)
try:
    validate_evaluation_protocol(protocol, policy_name)
    evaluation_preflight(
        source_checkpoint=checkpoint_path,
        requested_protocol=protocol,
        policy=policy_name,
    )
except (FileExistsError, ValueError) as error:
    parser.error(str(error))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Runtime imports must follow AppLauncher.
import gymnasium as gym  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA  # noqa: E402
from alexdoor_xas.adapters.a2 import A2Adapter  # noqa: E402
from alexdoor_xas.adapters.a3 import A3Adapter  # noqa: E402
from alexdoor_xas.adapters.limits import limits_for_robot  # noqa: E402
from alexdoor_xas.adapters.rollout import read_door_frame, rollout_chunks  # noqa: E402
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef  # noqa: E402
from alexdoor_xas.data_engine import apply_start_offset, plan_randomized_seeds  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.policies.act.policy import ActPolicy, act_chunk_source  # noqa: E402
from alexdoor_xas.policies.common.closed_loop import (  # noqa: E402
    closed_loop_trace_payload,
    factual_rollout_row,
    prepare_evaluation_run,
    protocol_rollouts,
    publish_closed_loop,
    rollout_key,
)
from alexdoor_xas.policies.diffusion.policy import (  # noqa: E402
    DiffusionPolicy,
    diffusion_chunk_source,
)
from alexdoor_xas.policies.scripted import alex_v2_variation_bounds  # noqa: E402


def _run() -> int:
    if os.environ.get("WANDB_MODE", "disabled").strip().lower() == "disabled":
        return _evaluate()

    os.environ.setdefault("WANDB_PROJECT", "alexdoor-xas")
    os.environ.setdefault("WANDB_NAME", f"{source_resolved['run_id']}-eval")
    os.environ.setdefault("WANDB_RUN_GROUP", policy_name)
    os.environ.setdefault("WANDB_JOB_TYPE", "eval")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError('W&B tracking requires: pip install -e ".[tracking]"') from error

    with wandb.init(
        dir=str(paths.OUTPUTS_DIR),
        config={
            "policy": policy_name,
            "source_run_id": source_resolved["run_id"],
            "dataset": source_resolved["config"]["dataset"],
            "evaluation": {
                "rollout_count": protocol["rollout_count"],
                "success_threshold_deg": protocol["success_threshold_deg"],
                "force_limit_n": protocol["force_limit_n"],
                "horizon_ticks": protocol["horizon_ticks"],
                "policy_execution": protocol["policy_execution"],
            },
        },
    ) as run:
        return _evaluate(run)


def _make_env(pose_id: str):
    control = protocol["control"]
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = 0
    cfg.sim.device = args.device
    cfg.sim.dt = float(control["sim_dt_s"])
    cfg.decimation = int(control["decimation"])
    cfg.sim.render_interval = cfg.decimation
    cfg.max_pos_delta_m = float(control["max_position_delta_m"])
    cfg.max_rot_delta_rad = float(control["max_rotation_delta_rad"])
    cfg.episode_length_s = float(protocol["horizon_ticks"]) * cfg.sim.dt * cfg.decimation
    cfg.door_pose_id = pose_id
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _fresh_adapter(action_space: str, env):
    center_w = env.shoulder_position_world_m()[0].detach().cpu().numpy()
    limits = limits_for_robot(
        ALEX_V2_ROBOT_TAG,
        calibration=env.alex_v2_calibration(),
        workspace_center_w=center_w,
    )
    a2 = A2Adapter(
        limits,
        contact_entry_shaping=bool(protocol["control"]["contact_entry_shaping"]),
    )
    if action_space == A2_EE_DELTA:
        return a2
    if action_space == A3_OBJ_REL_EE_DELTA:
        return A3Adapter(a2)
    raise ValueError(f"no adapter path for action space {action_space!r}")


def _load_policy(runtime_asset: RobotAssetRef, execution: dict):
    if policy_name == "act":
        policy = ActPolicy.from_checkpoint(
            checkpoint_path,
            device=policy_cfg.rollout.policy_device,
            runtime_asset=runtime_asset,
        )
    else:
        policy = DiffusionPolicy.from_checkpoint(
            checkpoint_path,
            device=policy_cfg.rollout.policy_device,
            sampler=str(execution["sampler"]),
            num_inference_steps=int(execution["num_inference_steps"]),
            runtime_asset=runtime_asset,
        )
    if policy.action_space != policy_cfg.dataset.space:
        raise RuntimeError("checkpoint action space differs from resolved config")
    return policy


def _chunk_source(policy, env, execution: dict, seed: int):
    if policy_name == "act":
        return act_chunk_source(
            policy,
            env,
            temporal_ensemble=bool(execution["temporal_ensemble"]),
            ensemble_m=float(execution["ensemble_m"]),
        )
    policy.seed(seed)
    return diffusion_chunk_source(
        policy,
        env,
        n_action_steps=int(execution["n_action_steps"]),
    )


def _evaluate(run=None) -> int:
    rows: list[dict] = []
    force_samples: dict[str, list[float]] = {}
    trace_payloads: dict[str, dict] = {}
    policy = None
    rollout_plan = protocol_rollouts(protocol)
    execution = protocol["policy_execution"]
    success_angle = math.radians(float(protocol["success_threshold_deg"]))

    for pose_spec in protocol["poses"]:
        pose_id = str(pose_spec["pose"])
        env = _make_env(pose_id)
        try:
            if policy is None:
                runtime_asset = RobotAssetRef.from_dict(env.robot_asset_provenance())
                policy = _load_policy(runtime_asset, execution)
            variations = {
                item.seed: item.variation
                for item in plan_randomized_seeds(
                    pose_spec["randomized_seeds"],
                    alex_v2_variation_bounds(env.alex_v2_calibration()),
                )
            }
            pose_items = [item for item in rollout_plan if item["pose"] == pose_id]
            for item in pose_items:
                seed = int(item["seed"])
                env.reset(seed=seed)
                variation = variations.get(seed) if item["status"] == "randomized" else None
                if variation is not None:
                    apply_start_offset(env, read_door_frame(env), variation)
                result = rollout_chunks(
                    env,
                    _chunk_source(policy, env, execution, seed),
                    _fresh_adapter(policy.action_space, env),
                    max_ticks=int(protocol["horizon_ticks"]),
                    stop_on_reject=bool(protocol["control"]["stop_on_reject"]),
                    success_angle_rad=success_angle,
                )
                row, forces = factual_rollout_row(
                    pose=pose_id,
                    seed=seed,
                    status=item["status"],
                    result=result,
                    control_dt_s=float(env.cfg.sim.dt) * int(env.cfg.decimation),
                    force_limit_n=float(protocol["force_limit_n"]),
                )
                key = rollout_key(pose_id, seed, item["status"])
                rows.append(row)
                force_samples[key] = forces
                trace_payloads[key] = closed_loop_trace_payload(result)
                print(
                    f"[eval_policy:{policy_name}] {key}: success={row['success']} "
                    f"reason={row['termination_reason']} steps={row['evaluated_steps']}",
                    flush=True,
                )
        finally:
            env.close()

    run_dir, resolved, source_publish = prepare_evaluation_run(
        source_checkpoint=checkpoint_path,
        requested_protocol=protocol,
        policy=policy_name,
        output_root=None,
    )
    metrics = publish_closed_loop(
        run_dir=run_dir,
        resolved=resolved,
        rows=rows,
        force_samples=force_samples,
        source_run_publish=source_publish,
        trace_payloads=trace_payloads,
        selected_trace_keys=set(args.trace_rollout),
    )
    overall = metrics["aggregate"]["overall"]
    if run is not None:
        run.log(
            {
                "closed_loop/rollout_count": overall["rollout_count"],
                "closed_loop/success_count": overall["success_count"],
                "closed_loop/success_rate": overall["success_rate"],
                "closed_loop/time_to_success_median_s": overall["time_to_success_s"]["median"],
                "closed_loop/time_to_success_p90_s": overall["time_to_success_s"]["p90"],
                "closed_loop/contact_force_mean_n": overall["contact_force_n"]["mean"],
                "closed_loop/contact_force_p95_n": overall["contact_force_n"]["p95"],
                "closed_loop/contact_force_maximum_n": overall["contact_force_n"]["maximum"],
                "closed_loop/impulse_ns": overall["impulse_ns"],
                "closed_loop/force_limit_exceedance_count": overall["force_limit_exceedance_count"],
                "closed_loop/adapter_accepted_rate": overall["adapter"]["accepted_rate"],
                "closed_loop/adapter_corrected_rate": overall["adapter"]["corrected_rate"],
                "closed_loop/adapter_rejected_rate": overall["adapter"]["rejected_rate"],
            }
        )
    print(
        f"[eval_policy:{policy_name}] "
        f"{overall['success_count']}/{overall['rollout_count']} successful; "
        f"published {run_dir / 'closed_loop'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = 0
    try:
        rc = _run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    finally:
        if args.clean_shutdown:
            simulation_app.close()
    sys.exit(rc)
