"""Negative fixtures for the smoke-summary gates (post-3.3 review WP5/WP8).

Each invariant family gets a fixture that breaks exactly one thing; the three
statuses (metadata coverage, protocol consistency, safety readiness) must
react independently.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from alexdoor_xas.policies.common.rollout_eval import trace_payload_hash

EXPECTED_POSES = {"D0", "D1"}
POSE_PLAN = {
    "poses": [
        {"pose_id": "D0", "door_yaw_deg": 0.0, "door_offset_x_m": 0.0, "door_offset_y_m": 0.0},
        {"pose_id": "D1", "door_yaw_deg": 2.86, "door_offset_x_m": 0.02, "door_offset_y_m": 0.0},
    ]
}
SEED_PLAN = {
    "poses": {
        "D0": {"base_seed": 100, "episodes_fixed": 1, "episodes_randomized": 1},
        "D1": {"base_seed": 100, "episodes_fixed": 1, "episodes_randomized": 1},
    }
}


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_smoke_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_smoke_eval_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(seed: int, randomized: bool, pose: dict) -> dict:
    return {
        "seed": seed,
        "randomized": randomized,
        "door_pose_id": pose["door_pose_id"],
        "door_yaw_deg": pose["door_yaw_deg"],
        "door_offset_xy": pose["door_offset_xy"],
        "success": True,
        "failure_label": None,
        "termination_reason": "success",
        "first_success_tick": 300,
        "time_to_success_s": 5.0,
        "env_truncated": False,
        "start_pose_settle": None,
        "initial_angle_rad": 0.0,
        "final_angle_rad": 0.8,
        "door_angle_change_rad": 0.8,
        "n_ticks": 300,
        "contact_ticks": 120,
        "contact_source": "force_sensor",
        "force_n": {"mean": 40.0, "max": 90.0, "p95": 80.0},
        "force_n_all_samples": {"max": 90.0, "n_exceedance_ticks": 0},
        "impulse_ns": 30.0,
        "contact_unavailable_reason": None,
        "force_exceeds_admission_bound": False,
        "force_trace_evidence": {
            "trace_sha256": "a" * 64,
            "admission_bound_n": 200.0,
            "peak_tick": 1,
            "peak_force_n": 90.0,
            "peak_contact": True,
            "peak_status": "accepted",
            "peak_requested": [0.0] * 6,
            "peak_applied": [0.0] * 6,
            "n_exceedance_ticks": 0,
            "exceedance_ticks": [],
            "window": [],
        },
        "n_accepted": 300,
        "n_corrected": 0,
        "n_rejected": 0,
        "n_warnings": 0,
        "warning_counts": {},
        "warning_family_counts": {},
        "warning_records": [],
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


def _payload(pose_id: str, plan_pose: dict) -> dict:
    pose = {
        "door_pose_id": pose_id,
        "door_yaw_deg": plan_pose["door_yaw_deg"],
        "door_offset_xy": [plan_pose["door_offset_x_m"], plan_pose["door_offset_y_m"]],
    }
    payload = {
        "policy": "act",
        "checkpoint": "outputs/act/best.pt",
        "checkpoint_sha256": "c" * 64,
        "robot_compatibility_label": "same_asset",
        "action_space": "A2_ee_delta",
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
        "door_pose": pose,
        "control_dt": 1.0 / 60.0,
        "dataset_provenance": {
            "dataset": {
                "task": "door_push_alex_v2",
                "space": "A2_ee_delta",
                "version": "v2_pose",
                "obs_preset": "core_door_pose",
            },
            "source_fingerprint_sha256": "s" * 64,
            "checkpoint_dataset_fingerprint_sha256": "e" * 64,
            "live_dataset_fingerprint_sha256": "e" * 64,
            "split_fingerprint_sha256": "f" * 64,
            "checkpoint_split_fingerprint_sha256": "f" * 64,
            "split_episode_ids": {
                "train": ["ep000", "ep001"],
                "val": ["ep002"],
                "test": ["ep003"],
            },
            "dataset_fingerprint_match": True,
            "split_fingerprint_match": True,
            "train_split_match": True,
            "val_split_match": True,
        },
        "seed_protocol": {
            "base_seed": 100,
            "episodes_fixed": 1,
            "episodes_randomized": 1,
            "fixed_seeds": [100],
            "randomized_seeds": [101],
            "variation_bounds": {},
        },
        "determinism_probe": {
            "kind": "repeat_same_seed_fresh_process",
            "seed": 100,
            "repeats": 2,
            "tolerances": {
                "command_abs": 0.0,
                "angle_abs_rad": 0.0,
                "force_abs_n": 0.0,
                "ee_position_abs_m": 0.0,
                "ee_orientation_abs": 0.0,
            },
            "trace_sha256": [],
            "reference_traces": {
                "n_ticks": 300,
                "requested": [[0.0] * 6] * 300,
                "applied": [[0.0] * 6] * 300,
                "statuses": ["accepted"] * 300,
                "reasons": [""] * 300,
                "warning_identities": [[] for _ in range(300)],
                "warning_family_counts": {},
                "first_success_tick": 300,
                "termination_reason": "success",
                "final_angle_rad": 0.8,
                "final_ee_position_world_m": [0.5, 0.0, 1.0],
                "final_ee_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "contact": [True] * 300,
                "force": [40.0] * 300,
            },
            "repeat_traces": [],
            "max_abs_diffs": {
                "requested": 0.0,
                "applied": 0.0,
                "final_angle_rad": 0.0,
                "force_n": 0.0,
                "final_ee_position_world_m": 0.0,
                "final_ee_orientation_world_xyzw": 0.0,
            },
            "mismatches": [],
            "passed": True,
        },
        "rollouts": [_row(100, False, pose), _row(101, True, pose)],
        "aggregate": {
            "n_rollouts": 2,
            "n_success": 2,
            "success_rate": 1.0,
            "fixed_reset_spread_rad": 0.0,
        },
    }
    payload["determinism_probe"]["trace_sha256"] = [
        trace_payload_hash(payload["determinism_probe"]["reference_traces"])
    ] * 2
    payload["determinism_probe"]["repeat_traces"] = [
        copy.deepcopy(payload["determinism_probe"]["reference_traces"]),
        copy.deepcopy(payload["determinism_probe"]["reference_traces"]),
    ]
    return payload


def _write_run(tmp_path: Path, name: str, payloads: dict[str, dict]) -> Path:
    run_dir = tmp_path / name
    (run_dir / "metrics").mkdir(parents=True)
    for pose_id, payload in payloads.items():
        (run_dir / "metrics" / f"act_eval_{pose_id}.json").write_text(json.dumps(payload))
    return run_dir


@pytest.fixture()
def valid_run(tmp_path):
    payloads = {
        pose["pose_id"]: _payload(pose["pose_id"], pose) for pose in POSE_PLAN["poses"]
    }
    return tmp_path, payloads


def _summarize(tmp_path, payloads, name="run_a"):
    cells = _matrix_payloads()
    cells.pop("act_a2")
    cells[name] = payloads
    reference = payloads[sorted(payloads)[0]]
    reference_provenance = reference.get("dataset_provenance") or {}
    reference_dataset = reference_provenance.get("dataset") or {}
    reference_exact = reference_provenance.get("checkpoint_dataset_fingerprint_sha256")
    for cell_name, cell_payloads in cells.items():
        if cell_name == name:
            continue
        for payload in cell_payloads.values():
            provenance = payload["dataset_provenance"]
            payload["obs_preset"] = reference.get("obs_preset")
            for field in ("task", "version", "obs_preset"):
                provenance["dataset"][field] = reference_dataset.get(field)
            for field in (
                "source_fingerprint_sha256",
                "split_fingerprint_sha256",
                "checkpoint_split_fingerprint_sha256",
                "split_episode_ids",
            ):
                provenance[field] = copy.deepcopy(reference_provenance.get(field))
            if payload["action_space"] == "A2_ee_delta":
                provenance["checkpoint_dataset_fingerprint_sha256"] = reference_exact
                provenance["live_dataset_fingerprint_sha256"] = reference_exact
    run_dirs = _write_matrix(tmp_path, cells)
    return _module().summarize(run_dirs, EXPECTED_POSES, POSE_PLAN, SEED_PLAN)


def _matrix_payloads() -> dict[str, dict[str, dict]]:
    cells: dict[str, dict[str, dict]] = {}
    for policy, space_tag, action_space, exact_fingerprint in (
        ("act", "a2", "A2_ee_delta", "2" * 64),
        ("act", "a3", "A3_obj_rel_ee_delta", "3" * 64),
        ("diffusion", "a2", "A2_ee_delta", "2" * 64),
        ("diffusion", "a3", "A3_obj_rel_ee_delta", "3" * 64),
    ):
        cell_name = f"{policy}_{space_tag}"
        payloads = {
            pose["pose_id"]: _payload(pose["pose_id"], pose) for pose in POSE_PLAN["poses"]
        }
        for payload in payloads.values():
            payload["checkpoint"] = f"outputs/{cell_name}/best.pt"
            payload["checkpoint_sha256"] = cell_name[0] * 64
            payload["action_space"] = action_space
            provenance = payload["dataset_provenance"]
            provenance["dataset"]["space"] = action_space
            provenance["checkpoint_dataset_fingerprint_sha256"] = exact_fingerprint
            provenance["live_dataset_fingerprint_sha256"] = exact_fingerprint
            if policy == "diffusion":
                payload["policy"] = "diffusion"
                del payload["chunk_size"]
                del payload["temporal_ensemble"]
                del payload["ensemble_m"]
                del payload["execution_mode"]
                payload.update(
                    horizon=16,
                    checkpoint_horizon=16,
                    n_action_steps=8,
                    sampler="ddim",
                    num_inference_steps=10,
                    execution_mode="receding_horizon",
                )
                for row in payload["rollouts"]:
                    for key in ("chunk_size", "temporal_ensemble", "ensemble_m"):
                        del row[key]
                    row.update(
                        policy_metadata_keys=[
                            "horizon",
                            "checkpoint_horizon",
                            "n_action_steps",
                            "sampler",
                            "num_inference_steps",
                            "execution_mode",
                        ],
                        horizon=16,
                        checkpoint_horizon=16,
                        n_action_steps=8,
                        sampler="ddim",
                        num_inference_steps=10,
                        execution_mode="receding_horizon",
                    )
        cells[cell_name] = payloads
    return cells


def _write_matrix(tmp_path: Path, cells: dict[str, dict[str, dict]]) -> list[Path]:
    run_dirs: list[Path] = []
    for cell_name, payloads in cells.items():
        run_dir = tmp_path / cell_name
        (run_dir / "metrics").mkdir(parents=True)
        policy = "diffusion" if cell_name.startswith("diffusion") else "act"
        for pose_id, payload in payloads.items():
            path = run_dir / "metrics" / f"{policy}_eval_{pose_id}.json"
            path.write_text(json.dumps(payload))
        run_dirs.append(run_dir)
    return run_dirs


def _summarize_matrix(tmp_path: Path, cells: dict[str, dict[str, dict]]) -> dict:
    return _module().summarize(
        _write_matrix(tmp_path, cells), EXPECTED_POSES, pose_plan=None, seed_plan=None
    )


# ── all-green baseline ───────────────────────────────────────────────────────


def test_valid_run_passes_all_three_gates(valid_run) -> None:
    tmp_path, payloads = valid_run
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "PASS"
    assert summary["protocol_consistency"] == "PASS"
    assert summary["safety_readiness"] == "PASS"
    run = summary["runs"]["run_a"]
    assert run["overall"]["n_rollouts"] == 4
    assert run["overall"]["termination_reasons"] == ["success"]


# ── metadata coverage (schema only) ─────────────────────────────────────────


def test_missing_row_field_fails_coverage_only(valid_run) -> None:
    tmp_path, payloads = valid_run
    del payloads["D0"]["rollouts"][0]["termination_reason"]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any("termination_reason" in p for p in summary["coverage_problems"])


@pytest.mark.parametrize(
    "field",
    [
        "final_ee_position_world_m",
        "reasons",
        "warning_identities",
        "warning_family_counts",
    ],
)
def test_incomplete_determinism_trace_fails_coverage(valid_run, field) -> None:
    tmp_path, payloads = valid_run
    del payloads["D0"]["determinism_probe"]["reference_traces"][field]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any(field in problem for problem in summary["coverage_problems"])


@pytest.mark.parametrize(
    "cell,field,value",
    [
        ("act_a2", "chunk_size", 20),
        ("diffusion_a2", "n_action_steps", 4),
    ],
)
def test_row_policy_metadata_mismatch_fails_coverage(tmp_path, cell, field, value) -> None:
    cells = _matrix_payloads()
    cells[cell]["D0"]["rollouts"][0][field] = value
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["metadata_coverage"] == "FAIL"
    assert any(field in problem for problem in summary["coverage_problems"])


def test_missing_policy_fails_coverage(valid_run) -> None:
    tmp_path, payloads = valid_run
    del payloads["D0"]["policy"]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any("'policy'" in problem for problem in summary["coverage_problems"])


def test_missing_all_sample_force_summary_fails_coverage(valid_run) -> None:
    tmp_path, payloads = valid_run
    del payloads["D0"]["rollouts"][0]["force_n_all_samples"]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any("force_n_all_samples" in problem for problem in summary["coverage_problems"])


def test_missing_exact_fingerprint_fails_coverage(valid_run) -> None:
    tmp_path, payloads = valid_run
    del payloads["D1"]["dataset_provenance"]["checkpoint_dataset_fingerprint_sha256"]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any("checkpoint_dataset_fingerprint_sha256" in p for p in summary["coverage_problems"])


@pytest.mark.parametrize(
    "field_path",
    [
        ("dataset", "task"),
        ("dataset", "version"),
        ("dataset", "obs_preset"),
        ("split_episode_ids",),
    ],
)
def test_missing_cross_cell_identity_field_fails_coverage(valid_run, field_path) -> None:
    tmp_path, payloads = valid_run
    target = payloads["D0"]["dataset_provenance"]
    for key in field_path[:-1]:
        target = target[key]
    del target[field_path[-1]]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any(".".join(field_path) in problem for problem in summary["coverage_problems"])


@pytest.mark.parametrize(
    "field",
    [
        "dataset_fingerprint_match",
        "split_fingerprint_match",
        "train_split_match",
        "val_split_match",
    ],
)
def test_missing_provenance_match_flag_fails_coverage(valid_run, field) -> None:
    tmp_path, payloads = valid_run
    del payloads["D0"]["dataset_provenance"][field]
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "FAIL"
    assert any(field in problem for problem in summary["coverage_problems"])


# ── protocol consistency ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "d" * 64),
        ("checkpoint", "outputs/other/best.pt"),
        ("action_space", "A3_obj_rel_ee_delta"),
        ("obs_preset", "core"),
        ("max_ticks", 400),
        ("success_angle_deg", 30.0),
        ("success_semantics", "chunk_boundary"),
        ("control_dt", 1.0 / 30.0),
    ],
)
def test_mixed_top_level_fields_fail_protocol(valid_run, field, value) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"][field] = value
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(field in p for p in summary["protocol_problems"])


def test_mixed_policy_metadata_fails_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"]["chunk_size"] = 20
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("policy metadata" in p for p in summary["protocol_problems"])


def test_mixed_split_identity_fails_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"]["dataset_provenance"]["split_fingerprint_sha256"] = "0" * 64
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("dataset/split identity" in p for p in summary["protocol_problems"])


@pytest.mark.parametrize(
    "field",
    [
        "dataset_fingerprint_match",
        "split_fingerprint_match",
        "train_split_match",
        "val_split_match",
    ],
)
def test_false_provenance_match_flag_fails_protocol(valid_run, field) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["dataset_provenance"][field] = False
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(field in problem for problem in summary["protocol_problems"])


def test_checkpoint_and_live_dataset_fingerprints_must_match(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["dataset_provenance"]["live_dataset_fingerprint_sha256"] = "0" * 64
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("checkpoint/live dataset fingerprint" in p for p in summary["protocol_problems"])


def test_row_disagreeing_with_top_pose_fails_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][1]["door_pose_id"] = "D1"
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("disagrees with" in p for p in summary["protocol_problems"])


def test_pose_plan_mismatch_fails_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"]["door_pose"]["door_yaw_deg"] = 9.99
    for row in payloads["D1"]["rollouts"]:
        row["door_yaw_deg"] = 9.99
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("pose plan" in p for p in summary["protocol_problems"])


def test_duplicate_seeds_fail_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][1]["seed"] = 100
    payloads["D0"]["rollouts"][1]["randomized"] = False
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("duplicate rollout seeds" in p for p in summary["protocol_problems"])


def test_seeds_off_protocol_fail(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][1]["seed"] = 999
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("seed protocol" in p for p in summary["protocol_problems"])


def test_self_declared_seed_protocol_cannot_drift_from_plan(valid_run) -> None:
    tmp_path, payloads = valid_run
    payload = payloads["D0"]
    payload["base_seed"] = 500
    payload["seed_protocol"].update(
        base_seed=500,
        fixed_seeds=[500],
        randomized_seeds=[501],
    )
    payload["determinism_probe"]["seed"] = 500
    payload["rollouts"][0]["seed"] = 500
    payload["rollouts"][1]["seed"] = 501
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("does not match seed plan" in p for p in summary["protocol_problems"])


def test_duplicate_pose_files_fail_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    run_dir = _write_run(tmp_path, "run_a", payloads)
    extra = copy.deepcopy(payloads["D0"])
    (run_dir / "metrics" / "act_eval_D0_copy.json").write_text(json.dumps(extra))
    summary = _module().summarize([run_dir], EXPECTED_POSES, POSE_PLAN, SEED_PLAN)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("duplicate eval files" in p for p in summary["protocol_problems"])


def test_missing_pose_fails_protocol(valid_run) -> None:
    tmp_path, payloads = valid_run
    del payloads["D1"]
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("pose matrix incomplete" in p for p in summary["protocol_problems"])


def test_success_failure_label_inconsistency_fails(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][0]["failure_label"] = "timeout_no_success"  # success stays True
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("inconsistent with" in p for p in summary["protocol_problems"])


def test_matrix_protocol_mismatch_across_runs_fails(valid_run) -> None:
    tmp_path, payloads = valid_run
    run_a = _write_run(tmp_path, "run_a", payloads)
    other = copy.deepcopy(payloads)
    for payload in other.values():
        payload["success_angle_deg"] = 30.0
    run_b = _write_run(tmp_path, "run_b", other)
    summary = _module().summarize([run_a, run_b], EXPECTED_POSES, POSE_PLAN, SEED_PLAN)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "success_angle_deg mismatch across cells" in p
        for p in summary["protocol_problems"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_fingerprint_sha256", "0" * 64),
        ("split_fingerprint_sha256", "1" * 64),
    ],
)
def test_cross_cell_shared_fingerprint_mismatch_fails(tmp_path, field, value) -> None:
    cells = _matrix_payloads()
    for payload in cells["diffusion_a3"].values():
        payload["dataset_provenance"][field] = value
        if field == "split_fingerprint_sha256":
            payload["dataset_provenance"]["checkpoint_split_fingerprint_sha256"] = value
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "diffusion_a3" in problem
        and "metrics/diffusion_eval_D0.json" in problem
        and field in problem
        for problem in summary["protocol_problems"]
    )


def test_cross_cell_split_membership_mismatch_fails(tmp_path) -> None:
    cells = _matrix_payloads()
    for payload in cells["act_a3"].values():
        payload["dataset_provenance"]["split_episode_ids"]["train"] = ["ep999"]
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "act_a3" in problem and "split_episode_ids" in problem
        for problem in summary["protocol_problems"]
    )


@pytest.mark.parametrize("field,value", [("task", "other_task"), ("version", "v3")])
def test_cross_cell_dataset_identity_mismatch_fails(tmp_path, field, value) -> None:
    cells = _matrix_payloads()
    for payload in cells["diffusion_a2"].values():
        payload["dataset_provenance"]["dataset"][field] = value
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "diffusion_a2" in problem and f"dataset.{field}" in problem
        for problem in summary["protocol_problems"]
    )


def test_cross_cell_observation_preset_mismatch_fails(tmp_path) -> None:
    cells = _matrix_payloads()
    for payload in cells["act_a3"].values():
        payload["obs_preset"] = "core"
        payload["dataset_provenance"]["dataset"]["obs_preset"] = "core"
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "act_a3" in problem and "obs_preset" in problem
        for problem in summary["protocol_problems"]
    )


def test_cross_cell_seed_plan_mismatch_fails(tmp_path) -> None:
    cells = _matrix_payloads()
    for payload in cells["diffusion_a2"].values():
        payload["base_seed"] = 500
        payload["seed_protocol"].update(
            base_seed=500,
            fixed_seeds=[500],
            randomized_seeds=[501],
        )
        payload["determinism_probe"]["seed"] = 500
        payload["rollouts"][0].update(seed=500)
        payload["rollouts"][1].update(seed=501)
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "diffusion_a2" in problem and "seed_plan" in problem
        for problem in summary["protocol_problems"]
    )


def test_cross_cell_door_pose_plan_mismatch_fails(tmp_path) -> None:
    cells = _matrix_payloads()
    payload = cells["act_a2"]["D1"]
    payload["door_pose"]["door_yaw_deg"] = 9.99
    for row in payload["rollouts"]:
        row["door_yaw_deg"] = 9.99
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "act_a2" in problem and "door_pose_plan" in problem
        for problem in summary["protocol_problems"]
    )


@pytest.mark.parametrize(
    "cell,new_fingerprint,expected_relationship",
    [
        ("diffusion_a2", "4" * 64, "same-space"),
        ("act_a3", "2" * 64, "cross-space"),
    ],
)
def test_cross_cell_exact_dataset_fingerprint_relationship_fails(
    tmp_path, cell, new_fingerprint, expected_relationship
) -> None:
    cells = _matrix_payloads()
    for payload in cells[cell].values():
        provenance = payload["dataset_provenance"]
        provenance["checkpoint_dataset_fingerprint_sha256"] = new_fingerprint
        provenance["live_dataset_fingerprint_sha256"] = new_fingerprint
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        cell in problem and expected_relationship in problem
        for problem in summary["protocol_problems"]
    )


def test_cross_cell_action_space_pairing_must_be_exact(tmp_path) -> None:
    cells = _matrix_payloads()
    for payload in cells["act_a3"].values():
        payload["action_space"] = "A2_ee_delta"
        payload["dataset_provenance"]["dataset"]["space"] = "A2_ee_delta"
        payload["dataset_provenance"]["checkpoint_dataset_fingerprint_sha256"] = "2" * 64
        payload["dataset_provenance"]["live_dataset_fingerprint_sha256"] = "2" * 64
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "act_a3" in problem and "matrix.cells" in problem
        for problem in summary["protocol_problems"]
    )


def test_partial_matrix_fails_with_cell_and_file_diagnostics(valid_run) -> None:
    tmp_path, payloads = valid_run
    run_dir = _write_run(tmp_path, "act_a2_only", payloads)
    summary = _module().summarize([run_dir], EXPECTED_POSES, POSE_PLAN, SEED_PLAN)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        "matrix.cells" in problem
        and str(run_dir) in problem
        and "act_eval_D0.json" in problem
        for problem in summary["protocol_problems"]
    )


@pytest.mark.parametrize(
    "cell,field,value",
    [("act_a3", "chunk_size", 20), ("diffusion_a3", "num_inference_steps", 100)],
)
def test_cross_cell_policy_configuration_mismatch_fails(tmp_path, cell, field, value) -> None:
    cells = _matrix_payloads()
    for payload in cells[cell].values():
        payload[field] = value
        if field == "num_inference_steps":
            for row in payload["rollouts"]:
                row[field] = value
    summary = _summarize_matrix(tmp_path, cells)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(
        cell in problem and f"policy_config.{field}" in problem
        for problem in summary["protocol_problems"]
    )


# ── determinism evidence ─────────────────────────────────────────────────────


def test_missing_determinism_probe_fails(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["determinism_probe"] = None
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("determinism probe" in p for p in summary["protocol_problems"])


def test_failed_determinism_probe_fails(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"]["determinism_probe"]["passed"] = False
    payloads["D1"]["determinism_probe"]["mismatches"] = ["repeat 1: n_ticks 5 != 6"]
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("determinism probe not passed" in p for p in summary["protocol_problems"])


def test_pending_replay_probe_fails(valid_run) -> None:
    # A probe with only the in-process reference (no fresh-process replay yet)
    # is not determinism evidence.
    tmp_path, payloads = valid_run
    payloads["D1"]["determinism_probe"]["repeats"] = 1
    payloads["D1"]["determinism_probe"]["passed"] = None
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("fresh-process replay has not run" in p for p in summary["protocol_problems"])


def test_single_repeat_probe_rejected(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["determinism_probe"]["repeats"] = 1
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("repeats" in p for p in summary["protocol_problems"])


def test_across_seed_spread_is_not_determinism_evidence(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["determinism_probe"]["kind"] = "fixed_reset_spread"
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("repeat_same_seed_fresh_process" in p for p in summary["protocol_problems"])


def test_in_process_repeats_are_not_determinism_evidence(valid_run) -> None:
    # Same-seed repeats inside one sim process are history-dependent; the gate
    # only accepts the fresh-process replay kind.
    tmp_path, payloads = valid_run
    payloads["D0"]["determinism_probe"]["kind"] = "repeat_same_seed"
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda probe: probe.update(seed=999), "does not match base_seed"),
        (lambda probe: probe["trace_sha256"].pop(), "trace_sha256 count"),
        (lambda probe: probe["trace_sha256"].__setitem__(0, "0" * 64), "first trace hash"),
        (lambda probe: probe["tolerances"].pop("force_abs_n"), "force_abs_n"),
        (lambda probe: probe["max_abs_diffs"].update(force_n=float("nan")), "force_n"),
    ],
)
def test_malformed_determinism_evidence_fails_protocol(valid_run, mutation, expected) -> None:
    tmp_path, payloads = valid_run
    mutation(payloads["D0"]["determinism_probe"])
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any(expected in problem for problem in summary["protocol_problems"])


def test_replay_trace_hash_may_differ_within_physical_tolerances(valid_run) -> None:
    tmp_path, payloads = valid_run
    probe = payloads["D0"]["determinism_probe"]
    probe["tolerances"]["ee_position_abs_m"] = 1e-9
    probe["repeat_traces"][1]["final_ee_position_world_m"][0] += 5e-10
    probe["trace_sha256"][1] = trace_payload_hash(probe["repeat_traces"][1])
    probe["max_abs_diffs"]["final_ee_position_world_m"] = abs(
        probe["repeat_traces"][1]["final_ee_position_world_m"][0]
        - probe["reference_traces"]["final_ee_position_world_m"][0]
    )
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "PASS"


@pytest.mark.parametrize(
    "diff_key,tolerance_key",
    [
        ("requested", "command_abs"),
        ("applied", "command_abs"),
        ("final_angle_rad", "angle_abs_rad"),
        ("force_n", "force_abs_n"),
    ],
)
def test_determinism_max_diff_cannot_exceed_stored_tolerance(
    valid_run, diff_key, tolerance_key
) -> None:
    tmp_path, payloads = valid_run
    probe = payloads["D0"]["determinism_probe"]
    probe["tolerances"][tolerance_key] = 0.01
    probe["max_abs_diffs"][diff_key] = 0.02
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("exceeds stored tolerance" in p for p in summary["protocol_problems"])


# ── diagnostics separation ───────────────────────────────────────────────────


def test_diagnostics_stay_out_of_primary_aggregates(valid_run) -> None:
    tmp_path, payloads = valid_run
    diag = copy.deepcopy(payloads["D0"])
    diag["door_pose"]["door_pose_id"] = "D0diag-ddpm100"
    for row in diag["rollouts"]:
        row["door_pose_id"] = "D0diag-ddpm100"
        row["force_exceeds_admission_bound"] = True  # must not leak into safety
    diag["chunk_size"] = 999
    diag["dataset_provenance"]["source_fingerprint_sha256"] = "0" * 64
    payloads["D0diag-ddpm100"] = diag
    summary = _summarize(tmp_path, payloads)
    run = summary["runs"]["run_a"]
    assert "D0diag-ddpm100" in run["diagnostics"]
    assert "D0diag-ddpm100" not in run["poses"]
    assert run["overall"]["n_rollouts"] == 4  # diag rows not aggregated
    assert summary["protocol_consistency"] == "PASS"
    assert summary["safety_readiness"] == "PASS"


# ── safety readiness (separate from metadata) ────────────────────────────────


def test_force_exceedance_yields_review_required_not_hidden(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][1]["force_exceeds_admission_bound"] = True
    payloads["D0"]["rollouts"][1]["force_n"] = {"mean": 90.0, "max": 272.2, "p95": 200.0}
    payloads["D0"]["rollouts"][1]["force_n_all_samples"] = {
        "max": 272.2,
        "n_exceedance_ticks": 1,
    }
    evidence = payloads["D0"]["rollouts"][1]["force_trace_evidence"]
    evidence.update(peak_force_n=272.2, n_exceedance_ticks=1, exceedance_ticks=[17])
    summary = _summarize(tmp_path, payloads)
    assert summary["metadata_coverage"] == "PASS"  # coverage independent
    assert summary["protocol_consistency"] == "PASS"
    assert summary["safety_readiness"] == "REVIEW_REQUIRED"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "REVIEW_REQUIRED"
    assert any("272.2" in reason for reason in safety["review_reasons"])
    assert safety["counts"]["n_force_exceeds_admission_bound"] == 1


def test_force_trace_exceedance_cannot_be_hidden_by_unset_row_flag(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][1]
    row["force_exceeds_admission_bound"] = False
    row["force_trace_evidence"].update(
        peak_force_n=250.0,
        peak_contact=False,
        n_exceedance_ticks=1,
        exceedance_ticks=[17],
    )
    row["force_n_all_samples"] = {"max": 250.0, "n_exceedance_ticks": 1}
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert summary["safety_readiness"] == "REVIEW_REQUIRED"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "REVIEW_REQUIRED"
    assert safety["counts"]["n_force_exceeds_admission_bound"] == 1


def test_force_exceedance_requires_matching_trace_evidence(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][1]
    row["force_exceeds_admission_bound"] = True
    row["force_n"]["max"] = 230.0
    row["force_n_all_samples"] = {"max": 230.0, "n_exceedance_ticks": 1}
    row["force_trace_evidence"].update(
        peak_force_n=100.0,
        n_exceedance_ticks=0,
        exceedance_ticks=[],
    )
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("force trace evidence" in p for p in summary["protocol_problems"])


@pytest.mark.parametrize(
    "field",
    ["mean", "max", "p95"],
)
def test_non_finite_force_evidence_fails_readiness(valid_run, field) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][0]["force_n"][field] = float("nan")
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"
    assert any(
        "non-finite rollout force/contact evidence" in reason
        for reason in summary["runs"]["run_a"]["safety_readiness"]["fail_reasons"]
    )


def test_non_finite_all_sample_force_evidence_fails_readiness(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    row["force_n_all_samples"]["max"] = float("inf")
    row["force_trace_evidence"]["peak_force_n"] = float("inf")
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"
    assert any(
        "non-finite rollout force/contact evidence" in reason
        for reason in summary["runs"]["run_a"]["safety_readiness"]["fail_reasons"]
    )


@pytest.mark.parametrize("field", ["initial_angle_rad", "final_angle_rad"])
def test_non_finite_rollout_angle_fails_safety(valid_run, field) -> None:
    _, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    row[field] = float("nan")

    safety = _module().assess_safety("run_a", [row])

    assert safety["status"] == "FAIL"
    assert any(
        "non-finite rollout force/contact evidence" in reason
        for reason in safety["fail_reasons"]
    )


def _warning_record(family_id: str, **evidence) -> dict:
    return {
        "id": family_id,
        "message": "machine-readable warning fixture",
        "evidence": evidence,
    }


def _set_warning_records(row: dict, records: list[dict]) -> None:
    row["n_warnings"] = len(records)
    row["warning_counts"] = {"machine-readable warning fixture": len(records)}
    family_counts: dict[str, int] = {}
    for record in records:
        family_id = record["id"]
        family_counts[family_id] = family_counts.get(family_id, 0) + 1
    row["warning_family_counts"] = family_counts
    row["warning_records"] = records


def _velocity_warning(
    *,
    tick: int = 0,
    exceedance: float = 0.5,
    consecutive: int = 1,
    count: int = 1,
    phase: str = "pre_contact",
) -> dict:
    return _warning_record(
        "a2.joint_velocity_limit",
        joint_index=13,
        joint_name="LEFT_KNEE_Y",
        tick_index=tick,
        rollout_phase=phase,
        measured_velocity_rad_s=9.8,
        configured_limit_rad_s=9.3,
        exceedance_rad_s=exceedance,
        consecutive_ticks=consecutive,
        duration_ticks=consecutive,
        count=count,
    )


def test_unknown_warning_family_requires_review(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][0]
    _set_warning_records(row, [_warning_record("a9.future_warning", detail="new")])
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "REVIEW_REQUIRED"
    assert safety["warning_families"]["a9.future_warning"]["status"] == "REVIEW_REQUIRED"


def test_legacy_warning_without_structured_evidence_requires_review(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][0]
    row["n_warnings"] = 1
    row["warning_counts"] = {"legacy free-text warning": 1}
    row["warning_family_counts"] = {}
    row["warning_records"] = []
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "REVIEW_REQUIRED"
    assert any("warning_records has 0 event" in reason for reason in safety["review_reasons"])


def test_extreme_velocity_warning_cannot_pass(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][0]
    _set_warning_records(row, [_velocity_warning(exceedance=8.0)])
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] in {"FAIL", "REVIEW_REQUIRED"}
    assert safety["warning_families"]["a2.joint_velocity_limit"]["status"] != "PASS"


def test_sustained_velocity_warnings_cannot_pass(valid_run) -> None:
    tmp_path, payloads = valid_run
    records = [
        _velocity_warning(tick=tick, consecutive=tick + 1, count=tick + 1)
        for tick in range(8)
    ]
    for pose_id in ("D0", "D1"):
        for row in payloads[pose_id]["rollouts"]:
            _set_warning_records(row, copy.deepcopy(records))
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] in {"FAIL", "REVIEW_REQUIRED"}
    assert safety["warning_families"]["a2.joint_velocity_limit"]["status"] != "PASS"


def test_bounded_settle_velocity_transient_with_latched_contact_passes(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    warning = _velocity_warning(tick=20, consecutive=2, phase="contact")
    warning["evidence"]["configured_limit_rad_s"] = 9.300000190734863
    _set_warning_records(row, [warning])
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "PASS"
    family = safety["warning_families"]["a2.joint_velocity_limit"]
    assert family["status"] == "PASS"
    assert family["count"] == 1


@pytest.mark.parametrize("phase", ["pre_contact", "contact"])
def test_velocity_warning_after_settle_window_requires_review(valid_run, phase) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    _set_warning_records(row, [_velocity_warning(tick=21, phase=phase)])
    summary = _summarize(tmp_path, payloads)
    family = summary["runs"]["run_a"]["safety_readiness"]["warning_families"]
    assert family["a2.joint_velocity_limit"]["status"] == "REVIEW_REQUIRED"


def test_three_consecutive_velocity_ticks_require_review(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    _set_warning_records(row, [_velocity_warning(consecutive=3)])
    summary = _summarize(tmp_path, payloads)
    family = summary["runs"]["run_a"]["safety_readiness"]["warning_families"]
    assert family["a2.joint_velocity_limit"]["status"] == "REVIEW_REQUIRED"


def test_configured_velocity_limit_beyond_float32_rounding_requires_review(valid_run) -> None:
    tmp_path, payloads = valid_run
    warning = _velocity_warning()
    warning["evidence"]["configured_limit_rad_s"] = 9.5
    row = payloads["D0"]["rollouts"][0]
    _set_warning_records(row, [warning])
    summary = _summarize(tmp_path, payloads)
    family = summary["runs"]["run_a"]["safety_readiness"]["warning_families"]
    assert family["a2.joint_velocity_limit"]["status"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_index", 14),
        ("configured_limit_rad_s", 100.0),
        ("tick_index", 21),
        ("duration_ticks", 3),
        ("count", 0),
    ],
)
def test_velocity_evidence_outside_each_bound_requires_review(valid_run, field, value) -> None:
    tmp_path, payloads = valid_run
    warning = _velocity_warning()
    warning["evidence"][field] = value
    row = payloads["D0"]["rollouts"][0]
    _set_warning_records(row, [warning])
    summary = _summarize(tmp_path, payloads)
    family = summary["runs"]["run_a"]["safety_readiness"]["warning_families"]
    assert family["a2.joint_velocity_limit"]["status"] == "REVIEW_REQUIRED"


def test_warning_message_counts_must_match_structured_records(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][0]
    _set_warning_records(row, [_velocity_warning()])
    row["warning_counts"] = {"different human message": 1}
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "REVIEW_REQUIRED"
    assert any("warning_counts do not match" in reason for reason in safety["review_reasons"])


@pytest.mark.parametrize("family_id", ["adapter.non_finite_state", "adapter.invalid_frame"])
def test_unsafe_warning_family_fails_safety(valid_run, family_id) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][0]
    _set_warning_records(row, [_warning_record(family_id, tick_index=4)])
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["warning_families"][family_id]["status"] == "FAIL"


def test_mixed_warning_families_take_worst_outcome(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][0]
    _set_warning_records(
        row,
        [
            _velocity_warning(),
            _warning_record("a9.future_warning", detail="new"),
            _warning_record("adapter.invalid_frame", tick_index=4),
        ],
    )
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "FAIL"
    assert {
        family_id: result["status"]
        for family_id, result in safety["warning_families"].items()
    } == {
        "a2.joint_velocity_limit": "PASS",
        "a9.future_warning": "REVIEW_REQUIRED",
        "adapter.invalid_frame": "FAIL",
    }


def test_systematic_rejections_fail_safety(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][0]["n_rejected"] = 30  # 30/630 > 2%
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert any("systematic adapter rejections" in reason for reason in safety["fail_reasons"])


def test_rejection_failure_label_fails_safety(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D0"]["rollouts"][1]
    row["success"] = False
    row["failure_label"] = "stopped_on_rejection"
    row["termination_reason"] = "rejection_stop"
    row["n_rejected"] = 1
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"


@pytest.mark.parametrize("field", ["failure_label", "termination_reason"])
def test_invalid_simulator_state_fails_safety(valid_run, field) -> None:
    _, payloads = valid_run
    row = payloads["D0"]["rollouts"][1]
    row[field] = "invalid_simulator_state"

    safety = _module().assess_safety("run_a", [row])

    assert safety["status"] == "FAIL"
    assert any(
        "invalid simulator state" in reason and "101" in reason
        for reason in safety["fail_reasons"]
    )


def test_stray_rejection_is_review_only(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["rollouts"][0]["n_rejected"] = 1  # 1/601 < 2%, no failure label
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "REVIEW_REQUIRED"


def test_env_truncation_is_at_least_review(valid_run) -> None:
    tmp_path, payloads = valid_run
    row = payloads["D1"]["rollouts"][1]
    row["env_truncated"] = True
    row["success"] = False
    row["failure_label"] = "env_truncated"
    row["termination_reason"] = "env_truncated"
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "REVIEW_REQUIRED"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert any("truncation" in reason for reason in safety["review_reasons"])


def test_zero_warnings_pass_safety(valid_run) -> None:
    tmp_path, payloads = valid_run
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "PASS"
    assert safety["counts"]["n_warnings"] == 0
