"""Pure regression tests for the Phase 3 unified evaluation workflow."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from alexdoor_xas.eval.phase3_unified import (
    UnifiedEvalError,
    _copy_exact,
    _publish_exclusive_json,
    bootstrap_mean_interval,
    build_eval_command,
    load_plan,
    paired_comparison,
    validate_eval_payload,
    verify_immutable_inventory,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "configs" / "phase3_unified_eval.v1.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def test_plan_freezes_exact_16_cell_576_rollout_matrix() -> None:
    plan = load_plan(PLAN_PATH)
    assert len(plan.cells) == 16
    assert [cell.index for cell in plan.cells] == list(range(16))
    assert sum(pose.n_rollouts for pose in plan.poses) == 36
    assert len(plan.cells) * sum(pose.n_rollouts for pose in plan.poses) == 576
    assert {(cell.policy, cell.action_space, cell.dataset_size) for cell in plan.cells} == {
        (policy, space, size)
        for policy in ("act", "diffusion")
        for space in ("A2_ee_delta", "A3_obj_rel_ee_delta")
        for size in (50, 100, 250, 500)
    }
    for cell in plan.cells:
        assert cell.returned_checkpoint.is_relative_to(plan.return_root)
        assert cell.workspace_checkpoint.is_relative_to(plan.workspace_root)
        assert not cell.workspace_checkpoint.is_relative_to(plan.return_root)


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("runtime", "simulation_device"), "cuda", "runtime"),
        (("rollout", "max_ticks"), 599, "rollout"),
        (("policies", "diffusion", "num_inference_steps"), 20, "policy"),
        (("cells", 0, "checkpoint_sha256"), "bad", "SHA-256"),
        (("cells", 0, "workspace_checkpoint"), "outputs/cluster_sweep/best.pt", "mapping"),
    ],
)
def test_plan_rejects_protocol_or_path_drift(tmp_path, path, value, message) -> None:
    payload = json.loads(PLAN_PATH.read_text())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(payload))
    with pytest.raises(UnifiedEvalError, match=message):
        load_plan(config)


def test_commands_freeze_cpu_sim_cuda_policy_and_ddim10() -> None:
    plan = load_plan(PLAN_PATH)
    pose = plan.pose("D0")
    act = plan.cell("sweep_act_a2_n50_seed0")
    act_command = build_eval_command(plan, act, pose, act.workspace_checkpoint)
    assert act_command[act_command.index("--device") + 1] == "cpu"
    assert "rollout.policy_device=cuda" in act_command
    assert "model.chunk_size=40" in act_command
    assert "rollout.temporal_ensemble=false" in act_command
    assert "+wandb.mode=disabled" in act_command

    diffusion = plan.cell("sweep_diffusion_a3_n50_seed0")
    diffusion_command = build_eval_command(
        plan, diffusion, pose, diffusion.workspace_checkpoint
    )
    assert diffusion_command[diffusion_command.index("--sampler") + 1] == "ddim"
    assert diffusion_command[diffusion_command.index("--inference-steps") + 1] == "10"
    assert "model.horizon=16" in diffusion_command
    assert "rollout.n_action_steps=8" in diffusion_command


def test_copy_is_independent_resumable_and_refuses_conflict(tmp_path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "copy.bin"
    source.write_bytes(b"checkpoint")
    _copy_exact(source, destination)
    assert destination.read_bytes() == b"checkpoint"
    assert source.stat().st_ino != destination.stat().st_ino
    _copy_exact(source, destination)
    destination.write_bytes(b"conflict")
    with pytest.raises(UnifiedEvalError, match="conflicting workspace"):
        _copy_exact(source, destination)


def test_exclusive_publication_resumes_identical_and_rejects_overwrite(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    _publish_exclusive_json(path, {"value": 1})
    _publish_exclusive_json(path, {"value": 1})
    with pytest.raises(UnifiedEvalError, match="refusing to overwrite"):
        _publish_exclusive_json(path, {"value": 2})


def test_immutable_inventory_detects_added_or_modified_file(tmp_path) -> None:
    plan = load_plan(PLAN_PATH)
    returned = tmp_path / "returned"
    returned.mkdir()
    payload = returned / "payload.bin"
    payload.write_bytes(b"immutable")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    inventory = provenance / "returned_package.sha256"
    inventory.write_text(
        f"{hashlib.sha256(b'immutable').hexdigest()}  ./payload.bin\n"
    )
    sizes = provenance / "returned_package_inventory.tsv"
    sizes.write_text("payload.bin\t9\n")
    isolated = dataclasses.replace(
        plan,
        return_root=returned,
        inventory_hashes=inventory,
        inventory_sizes=sizes,
    )
    assert verify_immutable_inventory(isolated) == []
    (returned / "extra.txt").write_text("extra")
    assert "exact path inventory changed" in "; ".join(verify_immutable_inventory(isolated))


def _isolated_act_fixture(tmp_path):
    plan = load_plan(PLAN_PATH)
    original = plan.cell("sweep_act_a2_n50_seed0")
    workspace = tmp_path / "workspace"
    run = workspace / "runs" / original.run_id
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    training = {
        "master_dataset_fingerprint_sha256": SHA_A,
        "action_dataset_fingerprint_sha256": SHA_B,
        "view_id": original.view_id,
        "view_fingerprint_sha256": SHA_C,
        "split_fingerprint_sha256": SHA_D,
        "normalization_sha256": SHA_E,
        "normalization_fingerprint_sha256": SHA_F,
    }
    (run / "logs").mkdir()
    (run / "logs" / "train_log.json").write_text(
        json.dumps({"training_provenance": training})
    )
    cell = dataclasses.replace(
        original,
        workspace_checkpoint=checkpoint,
        checkpoint_sha256=SHA_A,
    )
    isolated = dataclasses.replace(plan, workspace_root=workspace, cells=(cell,))
    pose = isolated.pose("D0")
    row = {
        "seed": 100,
        "randomized": False,
        "door_pose_id": "D0",
        "door_yaw_deg": 0.0,
        "door_offset_xy": [0.0, 0.0],
        "success": True,
        "initial_angle_rad": 0.0,
        "final_angle_rad": 0.8,
        "door_angle_change_rad": 0.8,
        "n_ticks": 2,
        "first_success_tick": 2,
        "time_to_success_s": 2.0 / 60.0,
        "termination_reason": "success",
        "failure_label": None,
        "env_truncated": False,
        "n_accepted": 2,
        "n_corrected": 0,
        "n_rejected": 0,
        "n_warnings": 0,
        "warning_counts": {},
        "warning_family_counts": {},
        "warning_records": [],
        "contact_ticks": 1,
        "contact_source": "force_sensor",
        "contact_unavailable_reason": None,
        "force_n": {"mean": 1.0, "max": 2.0, "p95": 2.0},
        "force_n_all_samples": {"max": 2.0, "n_exceedance_ticks": 0},
        "force_trace_evidence": {
            "trace_sha256": SHA_B,
            "admission_bound_n": 200.0,
            "peak_force_n": 2.0,
            "n_exceedance_ticks": 0,
            "exceedance_ticks": [],
        },
        "force_exceeds_admission_bound": False,
        "impulse_ns": 0.05,
        "policy_metadata_keys": [
            "chunk_size",
            "checkpoint_horizon",
            "temporal_ensemble",
            "ensemble_m",
            "execution_mode",
        ],
        "chunk_size": 40,
        "checkpoint_horizon": 40,
        "temporal_ensemble": False,
        "ensemble_m": 0.01,
        "execution_mode": "chunk_execution",
        "notes": "",
    }
    provenance = {
        "dataset": {
            "task": "door_push_alex_v2",
            "space": original.action_space,
            "version": "v3_scale_master",
            "view_id": original.view_id,
            "obs_preset": "core_door_pose",
        },
        "checkpoint_dataset_fingerprint_sha256": SHA_B,
        "live_dataset_fingerprint_sha256": SHA_B,
        "master_dataset_fingerprint_sha256": SHA_A,
        "action_dataset_fingerprint_sha256": SHA_B,
        "checkpoint_split_fingerprint_sha256": SHA_D,
        "split_fingerprint_sha256": SHA_D,
        "view_id": original.view_id,
        "view_fingerprint_sha256": SHA_C,
        "normalization_sha256": SHA_E,
        "split_episode_ids": {"train": ["train"], "val": ["val"], "test": ["test"]},
        "dataset_fingerprint_match": True,
        "split_fingerprint_match": True,
        "train_split_match": True,
        "val_split_match": True,
        "val_split_checked": True,
    }
    payload = {
        "policy": "act",
        "checkpoint": str(workspace / "staged" / "checkpoints" / "best.pt"),
        "checkpoint_sha256": SHA_A,
        "action_space": original.action_space,
        "obs_preset": "core_door_pose",
        "chunk_size": 40,
        "checkpoint_horizon": 40,
        "temporal_ensemble": False,
        "ensemble_m": 0.01,
        "execution_mode": "chunk_execution",
        "policy_device": "cuda",
        "max_ticks": 600,
        "success_angle_deg": 45.0,
        "success_semantics": "per_tick_first_crossing_stop",
        "base_seed": 100,
        "control_dt": 1.0 / 60.0,
        "door_pose": {"door_pose_id": "D0", "door_yaw_deg": 0.0, "door_offset_xy": [0, 0]},
        "seed_protocol": {
            "base_seed": 100,
            "episodes_fixed": 1,
            "episodes_randomized": 0,
            "fixed_seeds": [100],
            "randomized_seeds": [],
        },
        "dataset_provenance": provenance,
        "rollouts": [row],
        "aggregate": {"n_rollouts": 1, "n_success": 1},
    }
    return isolated, cell, pose, payload


def test_rollout_schema_accepts_nominal_preflight(tmp_path) -> None:
    plan, cell, pose, payload = _isolated_act_fixture(tmp_path)
    rows = validate_eval_payload(
        plan, cell, pose, payload, expected_fixed=1, expected_randomized=0
    )
    assert len(rows) == 1


@pytest.mark.parametrize(
    "mutator,message",
    [
        (lambda payload: payload["rollouts"][0].update(n_accepted=1), "every tick"),
        (lambda payload: payload["rollouts"][0].update(force_trace_evidence=None), "force"),
        (lambda payload: payload["rollouts"][0].update(seed=101), "seed plan"),
        (lambda payload: payload.update(checkpoint_horizon=39), "metadata"),
    ],
)
def test_rollout_schema_rejects_invalid_evidence(tmp_path, mutator, message) -> None:
    plan, cell, pose, payload = _isolated_act_fixture(tmp_path)
    broken = copy.deepcopy(payload)
    mutator(broken)
    with pytest.raises(UnifiedEvalError, match=message):
        validate_eval_payload(
            plan, cell, pose, broken, expected_fixed=1, expected_randomized=0
        )


def test_wilson_and_bootstrap_are_bounded_and_deterministic() -> None:
    low, high = wilson_interval(36, 36)
    assert 0.90 < low < high == 1.0
    assert wilson_interval(0, 36)[0] == 0.0
    first = bootstrap_mean_interval([1.0, 2.0, 3.0])
    second = bootstrap_mean_interval([1.0, 2.0, 3.0])
    assert first == second
    assert first is not None and first[0] <= 2.0 <= first[1]


def _comparison_rows(run_id: str, success_seed: int) -> list[dict]:
    rows = []
    for seed in range(36):
        rows.append(
            {
                "run_id": run_id,
                "door_pose_id": "D0",
                "seed": seed,
                "fixed_or_randomized": "fixed",
                "success": seed == success_seed,
                "final_angle_rad": float(seed),
                "total_ticks": 10,
                "contact_ticks": 2,
                "mean_contact_force_n": 3.0,
                "max_contact_force_n": 4.0,
                "p95_contact_force_n": 4.0,
                "contact_force_impulse_ns": 1.0,
                "adapter_corrected": 0,
                "adapter_rejected": 0,
                "adapter_warning_count": 0,
            }
        )
    return rows


def test_paired_comparison_uses_exact_36_key_alignment() -> None:
    left = _comparison_rows("left", success_seed=0)
    right = _comparison_rows("right", success_seed=1)
    comparison = paired_comparison(left, right, label="right minus left")
    assert comparison["n_pairs"] == 36
    assert comparison["success"]["right_wins"] == 1
    assert comparison["success"]["left_wins"] == 1
    right.pop()
    with pytest.raises(UnifiedEvalError, match="36 matched"):
        paired_comparison(left, right, label="incomplete")
