#!/usr/bin/env python
"""Closed-loop ACT evaluation through adapters on calibrated Alex V2.

Loads a trained ACT checkpoint, rebuilds the policy (norm stats embedded), and
rolls it out on ``AlexDoor-DoorPush-AlexV2-v0`` through the adapter matching the
checkpoint's action space (A2: world-frame deltas, A3: door-frame deltas).
Runs ``rollout.episodes_fixed`` fixed-reset rollouts (deterministic headless
physics makes this block a determinism probe) plus
``rollout.episodes_randomized`` rollouts with seeded EE start-offset
variations on held-out seeds. Writes per-rollout rows and aggregates
(success vs. the door-angle threshold, adapter accept/correct/reject/warning
counts) to ``metrics/act_eval.json`` next to the checkpoint. The legacy
scripted-baseline aggregate is embedded when ``rollout.reference_metrics`` is
set; ``rollout.matched_scripted_reference=true`` additionally runs the scripted
controller on the same fixed/randomized seed plan as ACT::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py \
        --viz none --device cuda:0 \
        rollout.policy_device=cuda \
        rollout.checkpoint=outputs/act_door_push/<run_id>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

# -- AppLauncher must be configured before any other Isaac import; the ACT
# config layer is torch/Isaac-free, so it resolves first (Hydra precedent).
from isaaclab.app import AppLauncher

from alexdoor_xas.policies.act import ActConfigError, load_act_config

parser = argparse.ArgumentParser(description="AlexDoor-XAS ACT closed-loop evaluation")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Trained ACT checkpoint (.pt) to evaluate."
)
parser.add_argument(
    "--reference-metrics",
    type=str,
    default=None,
    help="Scripted-baseline metrics.json to embed for comparison.",
)
parser.add_argument(
    "--matched-scripted-reference",
    action="store_true",
    default=None,
    help="Evaluate the scripted controller on the same rollout seed plan as ACT.",
)
parser.add_argument(
    "--determinism-replay",
    type=str,
    default=None,
    help=(
        "Path to an existing eval JSON: rerun its first fixed-seed rollout as this "
        "fresh process's first episode and complete the repeat-same-seed probe."
    ),
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

try:
    act_cfg = load_act_config(
        hydra_overrides,
        cli_overrides={
            "rollout.checkpoint": args.checkpoint,
            "rollout.reference_metrics": args.reference_metrics,
            "rollout.matched_scripted_reference": args.matched_scripted_reference,
        },
    )
except ActConfigError as error:
    parser.error(str(error))
if act_cfg.rollout.checkpoint is None:
    parser.error("rollout.checkpoint is required (--checkpoint or rollout.checkpoint=...)")
# A non-default door pose must carry an explicit pose label: rows and the
# per-pose metrics filename are keyed by it, so an unlabeled pose would be
# silently bucketed as the default pose in the smoke summary.
if act_cfg.rollout.door_pose_id is None and (
    act_cfg.rollout.door_yaw_deg != 0.0
    or act_cfg.rollout.door_offset_x != 0.0
    or act_cfg.rollout.door_offset_y != 0.0
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
from alexdoor_xas.policies.act.policy import (  # noqa: E402
    ActPolicy,
    act_chunk_source,
)
from alexdoor_xas.policies.act.rollout_eval import (  # noqa: E402
    aggregate_rollout_rows,
    contact_report,
    determinism_probe_reference,
    determinism_probe_update,
    rollout_failure_label,
    scripted_reference_payload,
    seed_protocol,
    summarize_decision_warnings,
)
from alexdoor_xas.policies.common.eval_metadata import (  # noqa: E402
    dataset_provenance,
    file_sha256,
    verify_checkpoint_dataset_binding,
)
from alexdoor_xas.policies.scripted import (  # noqa: E402
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)
from alexdoor_xas.tracking import load_wandb_config, start_wandb_run  # noqa: E402


def _make_env():
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = act_cfg.rollout.base_seed
    cfg.sim.device = args.device
    cfg.door_yaw_rad = math.radians(act_cfg.rollout.door_yaw_deg)
    cfg.door_offset_xy = (act_cfg.rollout.door_offset_x, act_cfg.rollout.door_offset_y)
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _door_pose_payload() -> dict:
    return {
        "door_pose_id": act_cfg.rollout.door_pose_id or "D0",
        "door_yaw_deg": act_cfg.rollout.door_yaw_deg,
        "door_offset_xy": [act_cfg.rollout.door_offset_x, act_cfg.rollout.door_offset_y],
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


def _run_rollout(
    env, policy, seed: int, variation, success_angle_rad: float
) -> tuple[dict, object]:
    env.reset(seed=seed)
    settle_report = None
    if variation is not None:
        settle_report = apply_start_offset(env, read_door_frame(env), variation)
    adapter = _fresh_adapter(policy.action_space, env)
    # Per-tick success semantics: the driver checks the hinge threshold after
    # every executed control tick and stops at the first crossing, so
    # first_success_tick is chunk-size independent (post-task extrapolation is
    # out of distribution and can knock the open door shut again).
    source = act_chunk_source(
        policy,
        env,
        temporal_ensemble=act_cfg.rollout.temporal_ensemble,
        ensemble_m=act_cfg.rollout.ensemble_m,
    )
    result = rollout_chunks(
        env,
        source,
        adapter,
        max_ticks=act_cfg.rollout.max_ticks,
        success_angle_rad=success_angle_rad,
    )
    warning_summary = summarize_decision_warnings(result.decisions_per_tick)
    control_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
    contact = contact_report(
        result.contact_per_tick,
        result.force_n_per_tick,
        control_dt,
        admission_bound_n=FORCE_DATASET_LIMIT_N,
    )
    success = bool(result.success)
    row = {
        "seed": seed,
        "randomized": variation is not None,
        **_door_pose_payload(),
        "success": success,
        "failure_label": rollout_failure_label(
            success=success,
            n_ticks=result.n_ticks,
            max_ticks=act_cfg.rollout.max_ticks,
            contact_ticks=contact["contact_ticks"],
            n_rejected=result.log.n_rejected,
            notes=result.notes,
            termination_reason=result.termination_reason,
        ),
        "termination_reason": result.termination_reason,
        "first_success_tick": result.first_success_tick,
        "time_to_success_s": (
            result.first_success_tick * control_dt
            if result.first_success_tick is not None
            else None
        ),
        "env_truncated": result.env_truncated,
        "start_pose_settle": settle_report,
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
    return row, result


def _reference_aggregate() -> dict | None:
    if act_cfg.rollout.reference_metrics is None:
        return None
    path = paths.REPO_ROOT / act_cfg.rollout.reference_metrics
    payload = json.loads(path.read_text())
    return {"path": str(path), "aggregate": payload.get("aggregate", payload)}


def _episode_plan(env):
    variation_bounds = alex_v2_variation_bounds(env.alex_v2_calibration())
    return plan_episodes(
        act_cfg.rollout.episodes_fixed,
        act_cfg.rollout.episodes_randomized,
        act_cfg.rollout.base_seed,
        variation_bounds,
    )


def _seed_protocol(env) -> dict:
    variation_bounds = alex_v2_variation_bounds(env.alex_v2_calibration())
    return seed_protocol(
        base_seed=act_cfg.rollout.base_seed,
        episodes_fixed=act_cfg.rollout.episodes_fixed,
        episodes_randomized=act_cfg.rollout.episodes_randomized,
        variation_bounds=variation_bounds,
    )


def _run_matched_scripted_reference(env, plan, success_angle_rad: float, protocol: dict) -> dict:
    engine_cfg = DataEngineCfg(
        task=paths.ALEX_V2_TASK,
        robot=ALEX_V2_ROBOT_TAG,
        success_angle_rad=success_angle_rad,
        max_ticks=act_cfg.rollout.max_ticks,
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
        checkpoint_path = paths.REPO_ROOT / act_cfg.rollout.checkpoint
        env = _make_env()
        runtime_asset = RobotAssetRef.from_dict(env.robot_asset_provenance())
        policy = ActPolicy.from_checkpoint(
            checkpoint_path,
            device=act_cfg.rollout.policy_device,
            runtime_asset=runtime_asset,
        )
        run_dir = checkpoint_path.parent.parent  # outputs/<experiment>/<run_id>/
        success_angle_rad = math.radians(act_cfg.rollout.success_angle_deg)
        print(
            f"[eval_act] checkpoint={checkpoint_path} space={policy.action_space} "
            f"obs={policy.obs_preset} chunk={policy.chunk_size} "
            f"ensemble={act_cfg.rollout.temporal_ensemble} "
            f"device={act_cfg.rollout.policy_device}",
            flush=True,
        )
        # Bind the eval to the exact trained dataset before any rollout: a
        # checkpoint/live fingerprint or split mismatch fails the evaluation.
        provenance = dataset_provenance(policy.checkpoint_config, run_dir, paths.DATASETS_DIR)
        provenance.update(
            verify_checkpoint_dataset_binding(policy.stats, provenance, paths.DATASETS_DIR)
        )
        if args.determinism_replay:
            return _run_determinism_replay(env, policy, checkpoint_path, success_angle_rad)

        plan = _episode_plan(env)
        protocol = _seed_protocol(env)
        rows: list[dict] = []
        first_fixed_result = None
        fixed_i = 0
        random_i = 0
        for item in plan:
            row, result = _run_rollout(env, policy, item.seed, item.variation, success_angle_rad)
            rows.append(row)
            if item.variation is None:
                if first_fixed_result is None:
                    first_fixed_result = result  # this process's first episode
                print(f"[fixed {fixed_i}] {_row_line(row)}", flush=True)
                fixed_i += 1
            else:
                print(f"[rand {random_i}] {_row_line(row)}", flush=True)
                random_i += 1

        # Repeat-same-seed determinism evidence: same-seed repeats *within* one
        # sim process are history-dependent (PhysX internal state evolves per
        # episode), so the probe records this process's first fixed rollout
        # and is completed by a fresh --determinism-replay process that reruns
        # it as *its* first episode with identical seeds/configuration.
        determinism_probe = None
        if first_fixed_result is not None:
            determinism_probe = determinism_probe_reference(
                first_fixed_result, seed=act_cfg.rollout.base_seed
            )
            print(
                f"[determinism] reference recorded (seed={act_cfg.rollout.base_seed}); "
                "fresh-process replay pending",
                flush=True,
            )

        aggregate = aggregate_rollout_rows(rows)
        matched_scripted_reference = None
        if act_cfg.rollout.matched_scripted_reference:
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
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "robot_compatibility_label": policy.robot_compatibility_label,
            "action_space": policy.action_space,
            "obs_preset": policy.obs_preset,
            "chunk_size": policy.chunk_size,
            "temporal_ensemble": act_cfg.rollout.temporal_ensemble,
            "policy_device": act_cfg.rollout.policy_device,
            "max_ticks": act_cfg.rollout.max_ticks,
            "success_angle_deg": act_cfg.rollout.success_angle_deg,
            "success_semantics": "per_tick_first_crossing_stop",
            "base_seed": act_cfg.rollout.base_seed,
            "door_pose": _door_pose_payload(),
            "control_dt": float(env.cfg.sim.dt) * int(env.cfg.decimation),
            "dataset_provenance": provenance,
            "seed_protocol": protocol,
            "determinism_probe": determinism_probe,
            "rollouts": rows,
            "aggregate": aggregate,
            "scripted_reference": _reference_aggregate(),
            "scripted_matched_reference": matched_scripted_reference,
        }
        # Pose-qualified filename so per-pose eval invocations never overwrite
        # each other (the default/no-pose-id eval keeps the frozen name).
        eval_name = (
            f"act_eval_{act_cfg.rollout.door_pose_id}.json"
            if act_cfg.rollout.door_pose_id
            else "act_eval.json"
        )
        metrics_path = run_dir / "metrics" / eval_name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

        wandb_cfg = load_wandb_config(
            overrides={
                "group": act_cfg.run.experiment,
                "name": f"{run_dir.name}_eval",
                "job_type": "eval",
                **act_cfg.wandb_overrides,
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
            f"[eval_act] success_rate={aggregate['success_rate']:.2f} "
            f"({aggregate['n_success']}/{aggregate['n_rollouts']}) "
            f"final_angle_mean={math.degrees(aggregate['final_angle_rad']['mean']):.1f} deg "
            f"adapter accepted/corrected/rejected="
            f"{aggregate['adapter']['n_accepted']}/{aggregate['adapter']['n_corrected']}/"
            f"{aggregate['adapter']['n_rejected']} "
            f"warnings={aggregate['adapter']['n_warnings']}",
            flush=True,
        )
        print(f"[eval_act] metrics: {metrics_path}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: ACT evaluation failed.", flush=True)
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


def _run_determinism_replay(env, policy, checkpoint_path, success_angle_rad: float) -> int:
    """Fresh-process leg of the repeat-same-seed probe: rerun + compare + record."""
    path = paths.REPO_ROOT / args.determinism_replay
    payload = json.loads(path.read_text())
    probe = payload.get("determinism_probe")
    if not probe:
        raise RuntimeError(f"{path} carries no determinism probe block to complete")
    # The replay is only valid evidence if this process is configured exactly
    # like the reference eval; any drift is a hard error, not a comparison.
    expected = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "action_space": policy.action_space,
        "obs_preset": policy.obs_preset,
        "max_ticks": act_cfg.rollout.max_ticks,
        "success_angle_deg": act_cfg.rollout.success_angle_deg,
        "base_seed": act_cfg.rollout.base_seed,
        "chunk_size": policy.chunk_size,
        "temporal_ensemble": act_cfg.rollout.temporal_ensemble,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"replay configuration mismatch on {key}: eval has "
                f"{payload.get(key)!r}, replay resolved {value!r}"
            )
    if payload.get("door_pose") != _door_pose_payload():
        raise RuntimeError(
            f"replay door pose {_door_pose_payload()} != eval {payload.get('door_pose')}"
        )
    _, result = _run_rollout(env, policy, act_cfg.rollout.base_seed, None, success_angle_rad)
    updated = determinism_probe_update(probe, result)
    payload["determinism_probe"] = updated
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[determinism-replay] seed={act_cfg.rollout.base_seed} "
        f"repeats={updated['repeats']} passed={updated['passed']} -> {path}",
        flush=True,
    )
    if not updated["passed"]:
        for mismatch in updated["mismatches"]:
            print(f"[determinism-mismatch] {mismatch}", flush=True)
        return 1
    return 0


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
