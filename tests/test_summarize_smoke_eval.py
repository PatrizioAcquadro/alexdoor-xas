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
        "n_warnings": 2,
        "warning_counts": {"arm joint JOINT_3 reached 4.4 rad/s after settle": 2},
        "notes": "",
    }


def _payload(pose_id: str, plan_pose: dict) -> dict:
    pose = {
        "door_pose_id": pose_id,
        "door_yaw_deg": plan_pose["door_yaw_deg"],
        "door_offset_xy": [plan_pose["door_offset_x_m"], plan_pose["door_offset_y_m"]],
    }
    payload = {
        "checkpoint": "outputs/act/best.pt",
        "checkpoint_sha256": "c" * 64,
        "robot_compatibility_label": "same_asset",
        "action_space": "A2_ee_delta",
        "obs_preset": "core_door_pose",
        "chunk_size": 40,
        "temporal_ensemble": False,
        "policy_device": "cuda",
        "max_ticks": 600,
        "success_angle_deg": 45.0,
        "success_semantics": "per_tick_first_crossing_stop",
        "base_seed": 100,
        "door_pose": pose,
        "control_dt": 1.0 / 60.0,
        "dataset_provenance": {
            "source_fingerprint_sha256": "s" * 64,
            "checkpoint_dataset_fingerprint_sha256": "e" * 64,
            "live_dataset_fingerprint_sha256": "e" * 64,
            "split_fingerprint_sha256": "f" * 64,
            "checkpoint_split_fingerprint_sha256": "f" * 64,
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
            "tolerances": {"command_abs": 0.0, "angle_abs_rad": 0.0, "force_abs_n": 0.0},
            "trace_sha256": [],
            "reference_traces": {
                "n_ticks": 300,
                "requested": [[0.0] * 6] * 300,
                "applied": [[0.0] * 6] * 300,
                "statuses": ["accepted"] * 300,
                "first_success_tick": 300,
                "termination_reason": "success",
                "final_angle_rad": 0.8,
                "contact": [True] * 300,
                "force": [40.0] * 300,
            },
            "max_abs_diffs": {
                "requested": 0.0,
                "applied": 0.0,
                "final_angle_rad": 0.0,
                "force_n": 0.0,
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
    run_dir = _write_run(tmp_path, name, payloads)
    return _module().summarize([run_dir], EXPECTED_POSES, POSE_PLAN, SEED_PLAN)


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
    assert any("across runs" in p for p in summary["protocol_problems"])


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


def test_every_replay_trace_hash_must_match_reference(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D0"]["determinism_probe"]["trace_sha256"][1] = "0" * 64
    summary = _summarize(tmp_path, payloads)
    assert summary["protocol_consistency"] == "FAIL"
    assert any("replay trace hash" in p for p in summary["protocol_problems"])


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
    payloads["D0diag-ddpm100"] = diag
    summary = _summarize(tmp_path, payloads)
    run = summary["runs"]["run_a"]
    assert "D0diag-ddpm100" in run["diagnostics"]
    assert "D0diag-ddpm100" not in run["poses"]
    assert run["overall"]["n_rollouts"] == 4  # diag rows not aggregated
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


def test_unsafe_warning_fails_safety(valid_run) -> None:
    tmp_path, payloads = valid_run
    payloads["D1"]["rollouts"][0]["warning_counts"] = {"non-finite command rejected": 1}
    payloads["D1"]["rollouts"][0]["n_warnings"] = 1
    summary = _summarize(tmp_path, payloads)
    assert summary["safety_readiness"] == "FAIL"
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert any("unsafe/invalid" in reason for reason in safety["fail_reasons"])


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


def test_benign_velocity_warnings_do_not_fail_safety(valid_run) -> None:
    tmp_path, payloads = valid_run
    summary = _summarize(tmp_path, payloads)
    safety = summary["runs"]["run_a"]["safety_readiness"]
    assert safety["status"] == "PASS"
    assert safety["counts"]["n_warnings"] == 8  # reported, never suppressed
