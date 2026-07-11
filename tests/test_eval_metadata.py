"""Eval-schema extensions: contact/force summaries, failure labels, provenance."""

from __future__ import annotations

import json

import pytest

from alexdoor_xas.policies.common.eval_metadata import dataset_provenance
from alexdoor_xas.policies.common.rollout_eval import contact_report, rollout_failure_label

CONTROL_DT = 1.0 / 60.0


def test_contact_report_force_math() -> None:
    contact = [False, True, True, True, False]
    forces = [0.0, 10.0, 30.0, 20.0, 0.0]
    report = contact_report(contact, forces, CONTROL_DT)
    assert report["contact_ticks"] == 3
    assert report["contact_source"] == "force_sensor"
    assert report["force_n"]["mean"] == pytest.approx(20.0)
    assert report["force_n"]["max"] == pytest.approx(30.0)
    assert report["force_n"]["p95"] == pytest.approx(30.0)  # nearest-rank on 3 samples
    assert report["impulse_ns"] == pytest.approx(60.0 * CONTROL_DT)
    assert report["unavailable_reason"] is None


def test_contact_report_without_force_sensing_records_reason() -> None:
    report = contact_report([None, None], [None, None], CONTROL_DT)
    assert report["contact_ticks"] == 0
    assert report["force_n"] is None
    assert report["impulse_ns"] is None
    assert "no contact force sensing" in report["unavailable_reason"]


def test_contact_report_contact_flag_without_any_contact() -> None:
    report = contact_report([False, False], [0.0, 0.0], CONTROL_DT)
    assert report["contact_ticks"] == 0
    assert report["force_n"] == {"mean": 0.0, "max": 0.0, "p95": 0.0}
    assert report["impulse_ns"] == pytest.approx(0.0)


def test_rollout_failure_labels_cover_buckets() -> None:
    common = {"n_ticks": 600, "max_ticks": 600, "contact_ticks": 5, "n_rejected": 0, "notes": ""}
    assert rollout_failure_label(success=True, **common) is None
    assert rollout_failure_label(success=False, **common) == "timeout_no_success"
    assert (
        rollout_failure_label(success=False, **common | {"contact_ticks": 0}) == "no_contact"
    )
    assert (
        rollout_failure_label(
            success=False, **common | {"notes": "stopped on rejected command: x"}
        )
        == "stopped_on_rejection"
    )
    assert (
        rollout_failure_label(success=False, **common | {"n_rejected": 3})
        == "commands_rejected"
    )
    # Rejection storm with zero motion: rejections outrank no_contact.
    assert (
        rollout_failure_label(success=False, **common | {"n_rejected": 3, "contact_ticks": 0})
        == "commands_rejected"
    )
    assert (
        rollout_failure_label(success=False, **common | {"n_ticks": 120})
        == "policy_stopped_early"
    )


def test_dataset_provenance_reads_manifest_splits_and_train_log(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    version_dir = datasets / "door_push_alex_v2" / "A2_ee_delta" / "v2_pose"
    version_dir.mkdir(parents=True)
    (version_dir / "manifest.json").write_text(
        json.dumps({"source_fingerprint_sha256": "abc123"})
    )
    splits_dir = datasets / "door_push_alex_v2" / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "v2_pose.json").write_text(
        json.dumps(
            {
                "n_episodes": 50,
                "fractions": [0.75, 0.125, 0.125],
                "seed": 0,
                "splits": {"train": ["a", "b"], "val": ["c"], "test": ["d"]},
            }
        )
    )
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "train_log.json").write_text(
        json.dumps(
            {"stats_source": "official", "train_episode_ids": ["a", "b"], "val_episode_ids": ["c"]}
        )
    )

    checkpoint_config = {
        "dataset": {
            "task": "door_push_alex_v2",
            "space": "A2_ee_delta",
            "version": "v2_pose",
            "obs_preset": "core_door_pose",
        }
    }
    out = dataset_provenance(checkpoint_config, run_dir, datasets)
    assert out["fingerprint_sha256"] == "abc123"
    assert out["split_summary"]["sizes"] == {"train": 2, "val": 1, "test": 1}
    assert out["split_episode_ids"]["test"] == ["d"]
    assert out["norm_stats_source"] == "official"
    assert out["train_log_split_ids"]["train"] == ["a", "b"]
    assert out["notes"] == []


def test_dataset_provenance_records_why_fields_are_missing(tmp_path) -> None:
    out = dataset_provenance(None, tmp_path / "nope", tmp_path / "datasets")
    assert out["fingerprint_sha256"] is None
    assert any("no embedded dataset config" in note for note in out["notes"])
    assert any("train_log.json" in note for note in out["notes"])


def test_rollout_cfg_accepts_door_pose_keys() -> None:
    from alexdoor_xas.policies.act.config import load_act_config
    from alexdoor_xas.policies.diffusion.config import load_diffusion_config

    act = load_act_config(
        [
            "rollout.door_yaw_deg=2.5",
            "rollout.door_offset_x=0.02",
            "rollout.door_pose_id=D3",
        ]
    )
    assert act.rollout.door_yaw_deg == pytest.approx(2.5)
    assert act.rollout.door_offset_x == pytest.approx(0.02)
    assert act.rollout.door_pose_id == "D3"

    dp = load_diffusion_config(
        ["rollout.door_yaw_deg=-2.5", "rollout.door_offset_y=-0.02"]
    )
    assert dp.rollout.door_yaw_deg == pytest.approx(-2.5)
    assert dp.rollout.door_offset_y == pytest.approx(-0.02)
    assert dp.rollout.door_pose_id is None


def test_contact_report_flags_admission_bound_exceedance() -> None:
    contact = [True, True]
    forces = [150.0, 250.0]
    flagged = contact_report(contact, forces, CONTROL_DT, admission_bound_n=200.0)
    assert flagged["force_exceeds_admission_bound"] is True
    ok = contact_report(contact, [150.0, 180.0], CONTROL_DT, admission_bound_n=200.0)
    assert ok["force_exceeds_admission_bound"] is False
    unbounded = contact_report(contact, forces, CONTROL_DT)
    assert unbounded["force_exceeds_admission_bound"] is None
    no_force = contact_report([None], [None], CONTROL_DT, admission_bound_n=200.0)
    assert no_force["force_exceeds_admission_bound"] is None
