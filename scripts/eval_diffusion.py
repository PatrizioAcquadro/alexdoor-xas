#!/usr/bin/env python
"""Closed-loop Diffusion Policy evaluation through adapters on calibrated Alex V2.

Loads a trained diffusion checkpoint, rebuilds the policy (norm stats and
noise schedule embedded), and rolls it out on ``AlexDoor-DoorPush-AlexV2-v0``
through the adapter matching the checkpoint's action space (A2: world-frame
deltas, A3: door-frame deltas). Execution is receding-horizon: each sampled
``Tp``-chunk contributes its first ``rollout.n_action_steps`` deltas before
the policy is re-queried. Sampling is seeded per rollout so the fixed-reset
block stays a determinism probe. Runs ``rollout.episodes_fixed`` fixed-reset
rollouts plus ``rollout.episodes_randomized`` rollouts with seeded EE
start-offset variations on held-out seeds. Writes per-rollout rows and
aggregates (success vs. the door-angle threshold, adapter
accept/correct/reject/warning counts) to ``metrics/diffusion_eval.json`` next
to the checkpoint. A reference metrics file (scripted baseline or an ACT
``act_eval.json``) is embedded when ``rollout.reference_metrics`` is set;
``rollout.matched_scripted_reference=true`` additionally runs the scripted
controller on the same fixed/randomized seed plan::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_diffusion.py \
        --viz none --device cuda:0 \
        rollout.policy_device=cuda \
        rollout.checkpoint=outputs/diffusion_door_push/<run_id>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

# -- AppLauncher must be configured before any other Isaac import; the
# diffusion config layer is torch/diffusers/Isaac-free, so it resolves first.
from isaaclab.app import AppLauncher

from alexdoor_xas.policies.diffusion import DiffusionConfigError, load_diffusion_config

parser = argparse.ArgumentParser(description="AlexDoor-XAS Diffusion Policy closed-loop eval")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Trained diffusion checkpoint (.pt)."
)
parser.add_argument(
    "--sampler", type=str, default=None, help="Rollout sampler: ddpm or ddim."
)
parser.add_argument(
    "--inference-steps", type=int, default=None, help="Denoising steps at rollout."
)
parser.add_argument(
    "--reference-metrics",
    type=str,
    default=None,
    help="Reference metrics json (scripted baseline or ACT act_eval.json) to embed.",
)
parser.add_argument(
    "--matched-scripted-reference",
    action="store_true",
    default=None,
    help="Evaluate the scripted controller on the same rollout seed plan.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

try:
    dp_cfg = load_diffusion_config(
        hydra_overrides,
        cli_overrides={
            "rollout.checkpoint": args.checkpoint,
            "rollout.sampler": args.sampler,
            "rollout.num_inference_steps": args.inference_steps,
            "rollout.reference_metrics": args.reference_metrics,
            "rollout.matched_scripted_reference": args.matched_scripted_reference,
        },
    )
except DiffusionConfigError as error:
    parser.error(str(error))
if dp_cfg.rollout.checkpoint is None:
    parser.error("rollout.checkpoint is required (--checkpoint or rollout.checkpoint=...)")
# A non-default door pose must carry an explicit pose label: rows and the
# per-pose metrics filename are keyed by it, so an unlabeled pose would be
# silently bucketed as the default pose in the smoke summary.
if dp_cfg.rollout.door_pose_id is None and (
    dp_cfg.rollout.door_yaw_deg != 0.0
    or dp_cfg.rollout.door_offset_x != 0.0
    or dp_cfg.rollout.door_offset_y != 0.0
):
    parser.error(
        "rollout.door_pose_id is required when a non-default door pose is set "
        "(rollout.door_yaw_deg / rollout.door_offset_x / rollout.door_offset_y)"
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA  # noqa: E402
from alexdoor_xas.adapters import (  # noqa: E402
    A2Adapter,
    A3Adapter,
    limits_for_robot,
    read_door_frame,
    rollout_chunks,
)
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef  # noqa: E402
from alexdoor_xas.data_engine import (  # noqa: E402
    DataEngineCfg,
    apply_start_offset,
    plan_episodes,
    run_episode,
)
from alexdoor_xas.envs.door_task.alex_v2_runtime import ALEX_V2_LIMITATIONS  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.eval.metrics import aggregate_metrics, episode_metrics  # noqa: E402
from alexdoor_xas.eval.sanity import FORCE_DATASET_LIMIT_N  # noqa: E402
from alexdoor_xas.policies.common.eval_metadata import dataset_provenance  # noqa: E402
from alexdoor_xas.policies.common.obs import stop_on_hinge_angle  # noqa: E402
from alexdoor_xas.policies.common.rollout_eval import (  # noqa: E402
    aggregate_rollout_rows,
    contact_report,
    rollout_failure_label,
    scripted_reference_payload,
    seed_protocol,
    summarize_decision_warnings,
)
from alexdoor_xas.policies.diffusion.policy import (  # noqa: E402
    DiffusionPolicy,
    diffusion_chunk_source,
)
from alexdoor_xas.policies.scripted import (  # noqa: E402
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)
from alexdoor_xas.tracking import load_wandb_config, start_wandb_run  # noqa: E402


def _make_env():
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = dp_cfg.rollout.base_seed
    cfg.sim.device = args.device
    cfg.door_yaw_rad = math.radians(dp_cfg.rollout.door_yaw_deg)
    cfg.door_offset_xy = (dp_cfg.rollout.door_offset_x, dp_cfg.rollout.door_offset_y)
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _door_pose_payload() -> dict:
    return {
        "door_pose_id": dp_cfg.rollout.door_pose_id or "D0",
        "door_yaw_deg": dp_cfg.rollout.door_yaw_deg,
        "door_offset_xy": [dp_cfg.rollout.door_offset_x, dp_cfg.rollout.door_offset_y],
    }


def _fresh_adapter(action_space: str, env):
    center_w = env.shoulder_position_world_m()[0].detach().cpu().numpy()
    limits = limits_for_robot(
        ALEX_V2_ROBOT_TAG,
        calibration=env.alex_v2_calibration(),
        workspace_center_w=center_w,
    )
    a2 = A2Adapter(limits)
    if action_space == A2_EE_DELTA:
        return a2
    if action_space == A3_OBJ_REL_EE_DELTA:
        return A3Adapter(a2)
    raise ValueError(f"no adapter path for action space {action_space!r}")


def _run_rollout(env, policy, seed: int, variation, success_angle_rad: float) -> dict:
    env.reset(seed=seed)
    if variation is not None:
        apply_start_offset(env, read_door_frame(env), variation)
    adapter = _fresh_adapter(policy.action_space, env)
    # Per-rollout sampling seed: the physics is deterministic headless, so a
    # seeded generator keeps the fixed-reset block a determinism probe.
    policy.seed(seed)
    # Rollouts end at the first chunk boundary past the success angle: the
    # demos terminate with the FSM, so post-task extrapolation is unbounded
    # (a wandering arm can knock the open door shut again).
    source = stop_on_hinge_angle(
        diffusion_chunk_source(policy, env, n_action_steps=dp_cfg.rollout.n_action_steps),
        success_angle_rad,
    )
    result = rollout_chunks(env, source, adapter, max_ticks=dp_cfg.rollout.max_ticks)
    warning_summary = summarize_decision_warnings(result.decisions_per_tick)
    control_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
    contact = contact_report(
        result.contact_per_tick,
        result.force_n_per_tick,
        control_dt,
        admission_bound_n=FORCE_DATASET_LIMIT_N,
    )
    success = bool(result.final_angle_rad >= success_angle_rad)
    return {
        "seed": seed,
        "randomized": variation is not None,
        **_door_pose_payload(),
        "sampler": dp_cfg.rollout.sampler,
        "num_inference_steps": dp_cfg.rollout.num_inference_steps,
        "success": success,
        "failure_label": rollout_failure_label(
            success=success,
            n_ticks=result.n_ticks,
            max_ticks=dp_cfg.rollout.max_ticks,
            contact_ticks=contact["contact_ticks"],
            n_rejected=result.log.n_rejected,
            notes=result.notes,
        ),
        "initial_angle_rad": result.initial_angle_rad,
        "final_angle_rad": result.final_angle_rad,
        "door_angle_change_rad": result.door_angle_change_rad,
        "n_ticks": result.n_ticks,
        "contact_ticks": contact["contact_ticks"],
        "contact_source": contact["contact_source"],
        "force_exceeds_admission_bound": contact["force_exceeds_admission_bound"],
        "force_n": contact["force_n"],
        "impulse_ns": contact["impulse_ns"],
        "contact_unavailable_reason": contact["unavailable_reason"],
        "n_accepted": result.log.n_accepted,
        "n_corrected": result.log.n_corrected,
        "n_rejected": result.log.n_rejected,
        "n_warnings": warning_summary["n_warnings"],
        "warning_counts": warning_summary["warning_counts"],
        "notes": result.notes,
    }


def _reference_aggregate() -> dict | None:
    if dp_cfg.rollout.reference_metrics is None:
        return None
    path = paths.REPO_ROOT / dp_cfg.rollout.reference_metrics
    payload = json.loads(path.read_text())
    return {"path": str(path), "aggregate": payload.get("aggregate", payload)}


def _episode_plan(env):
    variation_bounds = alex_v2_variation_bounds(env.alex_v2_calibration())
    return plan_episodes(
        dp_cfg.rollout.episodes_fixed,
        dp_cfg.rollout.episodes_randomized,
        dp_cfg.rollout.base_seed,
        variation_bounds,
    )


def _seed_protocol(env) -> dict:
    variation_bounds = alex_v2_variation_bounds(env.alex_v2_calibration())
    return seed_protocol(
        base_seed=dp_cfg.rollout.base_seed,
        episodes_fixed=dp_cfg.rollout.episodes_fixed,
        episodes_randomized=dp_cfg.rollout.episodes_randomized,
        variation_bounds=variation_bounds,
    )


def _run_matched_scripted_reference(env, plan, success_angle_rad: float, protocol: dict) -> dict:
    engine_cfg = DataEngineCfg(
        task=paths.ALEX_V2_TASK,
        robot=ALEX_V2_ROBOT_TAG,
        success_angle_rad=success_angle_rad,
        max_ticks=dp_cfg.rollout.max_ticks,
        limitations=ALEX_V2_LIMITATIONS,
    )
    controller_cfg = alex_v2_push_cfg(env.alex_v2_calibration())
    episodes = [
        run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
        for item in plan
    ]
    per_episode = [episode_metrics(episode) for episode in episodes]
    aggregate = aggregate_metrics(per_episode)
    return scripted_reference_payload(
        per_episode_metrics=per_episode,
        aggregate=aggregate,
        protocol=protocol,
    )


def main() -> int:
    rc = 0
    env = None
    try:
        checkpoint_path = paths.REPO_ROOT / dp_cfg.rollout.checkpoint
        env = _make_env()
        runtime_asset = RobotAssetRef.from_dict(env.robot_asset_provenance())
        policy = DiffusionPolicy.from_checkpoint(
            checkpoint_path,
            device=dp_cfg.rollout.policy_device,
            sampler=dp_cfg.rollout.sampler,
            num_inference_steps=dp_cfg.rollout.num_inference_steps,
            runtime_asset=runtime_asset,
        )
        run_dir = checkpoint_path.parent.parent  # outputs/<experiment>/<run_id>/
        # Sampler/horizon metadata must be consistent with the checkpoint: the
        # config-level Ta<=Tp guard ran against configs/diffusion.yaml's
        # horizon, which for a non-default checkpoint only matches when
        # model.horizon=<H> was passed — fail loudly instead of silently
        # evaluating with mislabeled horizon metadata.
        if policy.chunk_size != dp_cfg.model.horizon:
            raise RuntimeError(
                f"checkpoint horizon Tp={policy.chunk_size} != config model.horizon="
                f"{dp_cfg.model.horizon}; pass model.horizon={policy.chunk_size} so the "
                "recorded metadata and the Ta<=Tp check match the checkpoint"
            )
        if dp_cfg.rollout.n_action_steps > policy.chunk_size:
            raise RuntimeError(
                f"rollout.n_action_steps ({dp_cfg.rollout.n_action_steps}) exceeds the "
                f"checkpoint horizon Tp={policy.chunk_size}"
            )
        success_angle_rad = math.radians(dp_cfg.rollout.success_angle_deg)
        print(
            f"[eval_diffusion] checkpoint={checkpoint_path} space={policy.action_space} "
            f"obs={policy.obs_preset} Tp={policy.chunk_size} "
            f"Ta={dp_cfg.rollout.n_action_steps} sampler={dp_cfg.rollout.sampler}-"
            f"{dp_cfg.rollout.num_inference_steps} device={dp_cfg.rollout.policy_device}",
            flush=True,
        )
        plan = _episode_plan(env)
        protocol = _seed_protocol(env)
        rows: list[dict] = []
        fixed_i = 0
        random_i = 0
        for item in plan:
            row = _run_rollout(env, policy, item.seed, item.variation, success_angle_rad)
            rows.append(row)
            if item.variation is None:
                print(f"[fixed {fixed_i}] {_row_line(row)}", flush=True)
                fixed_i += 1
            else:
                print(f"[rand {random_i}] {_row_line(row)}", flush=True)
                random_i += 1

        aggregate = aggregate_rollout_rows(rows)
        matched_scripted_reference = None
        if dp_cfg.rollout.matched_scripted_reference:
            matched_scripted_reference = _run_matched_scripted_reference(
                env, plan, success_angle_rad, protocol
            )
            matched = matched_scripted_reference["aggregate"]
            print(
                f"[matched scripted] success_rate={matched['success_rate']:.2f} "
                f"({matched['n_success']}/{matched['n_episodes']}) "
                f"final_angle_mean="
                f"{math.degrees(matched['final_door_angle_rad']['mean']):.1f} deg",
                flush=True,
            )
        payload = {
            "checkpoint": str(checkpoint_path),
            "robot_compatibility_label": policy.robot_compatibility_label,
            "action_space": policy.action_space,
            "obs_preset": policy.obs_preset,
            "horizon": policy.chunk_size,
            "n_action_steps": dp_cfg.rollout.n_action_steps,
            "sampler": dp_cfg.rollout.sampler,
            "num_inference_steps": dp_cfg.rollout.num_inference_steps,
            "policy_device": dp_cfg.rollout.policy_device,
            "max_ticks": dp_cfg.rollout.max_ticks,
            "success_angle_deg": dp_cfg.rollout.success_angle_deg,
            "base_seed": dp_cfg.rollout.base_seed,
            "config_horizon": dp_cfg.model.horizon,
            "door_pose": _door_pose_payload(),
            "control_dt": float(env.cfg.sim.dt) * int(env.cfg.decimation),
            "dataset_provenance": dataset_provenance(
                policy.checkpoint_config, run_dir, paths.DATASETS_DIR
            ),
            "seed_protocol": protocol,
            "rollouts": rows,
            "aggregate": aggregate,
            "reference": _reference_aggregate(),
            "scripted_matched_reference": matched_scripted_reference,
        }
        eval_name = (
            f"diffusion_eval_{dp_cfg.rollout.door_pose_id}.json"
            if dp_cfg.rollout.door_pose_id
            else "diffusion_eval.json"
        )
        metrics_path = run_dir / "metrics" / eval_name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

        wandb_cfg = load_wandb_config(
            overrides={
                "group": dp_cfg.run.experiment,
                "name": f"{run_dir.name}_eval",
                "job_type": "eval",
                **dp_cfg.wandb_overrides,
            }
        )
        with start_wandb_run(wandb_cfg, config=payload | {"rollouts": None}) as run:
            run.log(
                {
                    "eval/success_rate": aggregate["success_rate"],
                    "eval/final_angle_mean_rad": aggregate["final_angle_rad"]["mean"],
                    "eval/n_corrected": aggregate["adapter"]["n_corrected"],
                    "eval/n_rejected": aggregate["adapter"]["n_rejected"],
                    "eval/n_warnings": aggregate["adapter"]["n_warnings"],
                }
            )

        print(
            f"[eval_diffusion] success_rate={aggregate['success_rate']:.2f} "
            f"({aggregate['n_success']}/{aggregate['n_rollouts']}) "
            f"final_angle_mean={math.degrees(aggregate['final_angle_rad']['mean']):.1f} deg "
            f"adapter accepted/corrected/rejected="
            f"{aggregate['adapter']['n_accepted']}/{aggregate['adapter']['n_corrected']}/"
            f"{aggregate['adapter']['n_rejected']} "
            f"warnings={aggregate['adapter']['n_warnings']}",
            flush=True,
        )
        print(f"[eval_diffusion] metrics: {metrics_path}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: Diffusion Policy evaluation failed.", flush=True)
        rc = 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
        if args.clean_shutdown:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
    return rc


def _row_line(row: dict) -> str:
    return (
        f"seed={row['seed']} success={row['success']} "
        f"final={math.degrees(row['final_angle_rad']):.1f} deg ticks={row['n_ticks']} "
        f"a/c/r={row['n_accepted']}/{row['n_corrected']}/{row['n_rejected']} "
        f"warnings={row['n_warnings']}"
    )


if __name__ == "__main__":
    # os._exit avoids Kit shutdown masking the exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
