"""Canonical learned-run allocation, frozen protocol, and artifact contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alexdoor_xas.policies.act.config import load_act_config
from alexdoor_xas.policies.common.runs import (
    allocate_run_directory,
    dataset_token,
    frozen_evaluation_protocol,
    load_resolved_config,
    resolve_resume_directory,
    resolved_training_config,
    write_json_atomic,
)


def test_dataset_token_compacts_view_and_uses_version_without_view() -> None:
    assert dataset_token("v2_pose", None) == "v2pose"
    assert dataset_token("v3_scale_master", "v3_scale_n500") == "v3n500"


def test_run_allocation_is_exclusive_and_collision_safe(tmp_path) -> None:
    now = datetime(2026, 8, 12, 15, 30, 45, tzinfo=UTC)
    first_id, first = allocate_run_directory(
        output_root=tmp_path,
        policy="act",
        action_space="A3_obj_rel_ee_delta",
        dataset_version="v3_scale_master",
        dataset_view_id="v3_scale_n500",
        seed=0,
        now=now,
    )
    second_id, second = allocate_run_directory(
        output_root=tmp_path,
        policy="act",
        action_space="A3_obj_rel_ee_delta",
        dataset_version="v3_scale_master",
        dataset_view_id="v3_scale_n500",
        seed=0,
        now=now,
    )
    assert first_id == "20260812T153045Z_a3_v3n500_seed0"
    assert second_id == first_id + "_r2"
    assert first.parent == second.parent == tmp_path / "door_push_alex_v2" / "act"
    assert first.is_dir() and second.is_dir()


def test_frozen_training_protocol_has_exact_36_rollouts() -> None:
    cfg = load_act_config()
    protocol = frozen_evaluation_protocol("act", cfg.rollout)
    assert protocol["rollout_count"] == 36
    by_pose = {entry["pose"]: entry for entry in protocol["poses"]}
    assert by_pose["D0"]["fixed_seeds"] == list(range(100, 105))
    assert by_pose["D0"]["randomized_seeds"] == list(range(105, 120))
    for pose, base in (("D1", 200), ("D2", 210), ("D3", 220), ("D4", 230)):
        assert by_pose[pose]["fixed_seeds"] == [base]
        assert by_pose[pose]["randomized_seeds"] == [base + 1, base + 2, base + 3]
    assert protocol["force_limit_n"] == 200.0


def test_resolved_config_is_immutable_and_resume_requires_last(tmp_path) -> None:
    cfg = load_act_config([f"run.output_root={tmp_path}"])
    run_id, run_dir = allocate_run_directory(
        output_root=tmp_path,
        policy="act",
        action_space=cfg.dataset.space,
        dataset_version=cfg.dataset.version,
        dataset_view_id=cfg.dataset.view_id,
        seed=cfg.train.seed,
    )
    payload = resolved_training_config(run_id=run_id, policy="act", config=cfg)
    write_json_atomic(run_dir / "resolved_config.json", payload, exclusive=True)
    with pytest.raises(FileExistsError, match="immutable"):
        write_json_atomic(run_dir / "resolved_config.json", payload, exclusive=True)
    with pytest.raises(ValueError, match="last.pt"):
        resolve_resume_directory(run_dir, "act")
    last = run_dir / "checkpoints" / "last.pt"
    last.parent.mkdir()
    last.write_bytes(b"resumable")
    assert resolve_resume_directory(run_dir, "act") == run_dir.resolve()
    assert load_resolved_config(run_dir)["run_id"] == run_id
