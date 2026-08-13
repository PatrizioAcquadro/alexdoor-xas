"""Closed-loop schemas, aggregation, routing, and optional-artifact contracts."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.policies.act.config import load_act_config
from alexdoor_xas.policies.common.closed_loop import (
    aggregate_closed_loop,
    closed_loop_trace_payload,
    factual_rollout_row,
    prepare_evaluation_run,
    protocol_rollouts,
    publish_closed_loop,
    rollout_key,
    validate_evaluation_protocol,
)
from alexdoor_xas.policies.common.runs import (
    RUN_FORMAT,
    frozen_evaluation_protocol,
    resolved_training_config,
    write_json_atomic,
)


def _result(
    *,
    success: bool,
    forces: list[float | None],
    accepted: int = 2,
    corrected: int = 1,
    rejected: int = 0,
    warning_ids: tuple[str, ...] = (),
):
    warning_records = tuple(SimpleNamespace(id=value) for value in warning_ids)
    decisions = [
        SimpleNamespace(
            requested=[0.01, 0, 0, 0, 0, 0],
            applied=[0.01, 0, 0, 0, 0, 0],
            status="accepted",
            warning_records=warning_records,
        )
    ]
    return SimpleNamespace(
        success=success,
        termination_reason="success" if success else "tick_budget",
        environment_terminated=False,
        environment_truncated=False,
        n_ticks=len(forces),
        first_success_tick=2 if success else None,
        force_n_per_tick=forces,
        contact_per_tick=[value is not None and value > 1.0 for value in forces],
        decisions_per_tick=decisions,
        log=SimpleNamespace(
            n_accepted=accepted,
            n_corrected=corrected,
            n_rejected=rejected,
        ),
    )


def _row(pose: str, seed: int, status: str, result):
    return factual_rollout_row(
        pose=pose,
        seed=seed,
        status=status,
        result=result,
        control_dt_s=1 / 60,
        force_limit_n=200.0,
    )


def _source_run(tmp_path: Path):
    cfg = load_act_config()
    run_dir = tmp_path / "door_push_alex_v2" / "act" / "source-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"self-contained")
    (run_dir / "checkpoints" / "last.pt").write_bytes(b"allowed")
    resolved = resolved_training_config(run_id="source-run", policy="act", config=cfg)
    write_json_atomic(run_dir / "resolved_config.json", resolved)
    return run_dir, resolved


def test_protocol_expands_exact_default_36_rollouts() -> None:
    protocol = frozen_evaluation_protocol("act", load_act_config().rollout)
    validate_evaluation_protocol(protocol, "act")
    items = protocol_rollouts(protocol)
    assert len(items) == 36
    assert sum(item["pose"] == "D0" and item["status"] == "fixed" for item in items) == 5
    assert sum(item["pose"] == "D0" and item["status"] == "randomized" for item in items) == 15
    assert all(
        sum(item["pose"] == pose for item in items) == 4 for pose in ("D1", "D2", "D3", "D4")
    )


def test_closed_loop_rows_and_aggregates_are_factual_and_compact() -> None:
    protocol = frozen_evaluation_protocol("act", load_act_config().rollout)
    rows = []
    forces = {}
    for index, item in enumerate(protocol_rollouts(protocol)):
        result = _result(
            success=index % 3 != 0,
            forces=[10.0 + index, 250.0 if index == 0 else 20.0 + index],
            warning_ids=("a2.joint_velocity_limit",) if index == 0 else (),
        )
        row, samples = _row(item["pose"], item["seed"], item["status"], result)
        rows.append(row)
        forces[row["rollout_key"]] = samples

    aggregate = aggregate_closed_loop(rows, forces)
    assert aggregate["overall"]["rollout_count"] == 36
    assert aggregate["overall"]["success_count"] == 24
    assert aggregate["overall"]["time_to_success_s"]["sample_count"] == 24
    assert aggregate["overall"]["contact_force_n"]["sample_count"] == 72
    assert aggregate["overall"]["force_limit_exceedance_count"] == 1
    assert aggregate["by_pose"]["D0"]["rollout_count"] == 20
    assert aggregate["by_status"]["fixed"]["rollout_count"] == 9
    assert aggregate["by_status"]["randomized"]["rollout_count"] == 27
    assert aggregate["by_pose_and_status"]["D3"]["randomized"]["rollout_count"] == 3
    serialized = json.dumps({"rollouts": rows, "aggregate": aggregate})
    for forbidden in (
        "failure_label",
        "final_angle",
        "fixed_reset_spread",
        "warning_records",
        "force_trace_evidence",
        "force_window",
    ):
        assert forbidden not in serialized
    assert rows[0]["warning_family_counts"] == {"a2.joint_velocity_limit": 1}
    assert rows[1]["termination_reason"] == "controller_done"


def test_evaluations_are_exclusive_children_with_minimal_preflight(tmp_path) -> None:
    source_run, resolved = _source_run(tmp_path)
    best = source_run / "checkpoints" / "best.pt"
    training_report = source_run / "report.md"
    training_report.write_text("training report\n")
    legacy = source_run / "closed_loop"
    legacy.mkdir()
    (legacy / "metrics.json").write_text("legacy\n")

    first, first_resolved = prepare_evaluation_run(
        source_checkpoint=best,
        requested_protocol=resolved["evaluation_protocol"],
        policy="act",
    )
    changed = deepcopy(resolved["evaluation_protocol"])
    changed["force_limit_n"] = 150.0
    second, second_resolved = prepare_evaluation_run(
        source_checkpoint=best,
        requested_protocol=changed,
        policy="act",
    )
    assert first.parent == second.parent == legacy
    assert first != second
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:_r\d+)?", first.name)
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:_r\d+)?", second.name)
    assert (legacy / "metrics.json").read_text() == "legacy\n"
    assert training_report.read_text() == "training report\n"
    assert set(path.name for path in source_run.parent.iterdir()) == {"source-run"}
    assert set(path.name for path in first.iterdir()) == {"resolved_config.json"}
    assert set(first_resolved) == {
        "format",
        "run_type",
        "run_id",
        "policy",
        "created_utc",
        "source_run_id",
        "checkpoint",
        "config",
        "evaluation_protocol",
    }
    assert first_resolved["format"] == RUN_FORMAT
    assert first_resolved["run_type"] == "evaluation"
    assert first_resolved["source_run_id"] == "source-run"
    assert first_resolved["checkpoint"] == str(best.resolve())
    assert second_resolved["evaluation_protocol"] == changed


def test_publish_writes_one_summary_and_only_required_traces(tmp_path) -> None:
    source_run, source_resolved = _source_run(tmp_path)
    changed = deepcopy(source_resolved["evaluation_protocol"])
    changed["poses"] = [deepcopy(changed["poses"][0])]
    changed["poses"][0]["fixed_seeds"] = [100, 101, 102]
    changed["poses"][0]["randomized_seeds"] = []
    changed["rollout_count"] = 3
    training_report = source_run / "report.md"
    training_report.write_text("training report\n")
    run_dir, resolved = prepare_evaluation_run(
        source_checkpoint=source_run / "checkpoints" / "best.pt",
        requested_protocol=changed,
        policy="act",
    )
    results = [
        _result(success=True, forces=[10.0, 20.0]),
        _result(success=False, forces=[10.0, 20.0]),
        _result(success=True, forces=[10.0, 220.0]),
    ]
    rows = []
    samples = {}
    traces = {}
    for seed, result in zip((100, 101, 102), results, strict=True):
        row, force_values = _row("D0", seed, "fixed", result)
        rows.append(row)
        samples[row["rollout_key"]] = force_values
        traces[row["rollout_key"]] = closed_loop_trace_payload(result)
    metrics = publish_closed_loop(
        run_dir=run_dir,
        resolved=resolved,
        rows=rows,
        force_samples=samples,
        trace_payloads=traces,
        selected_trace_keys={rollout_key("D0", 100, "fixed")},
    )
    assert metrics["aggregate"]["overall"]["success_count"] == 2
    assert set(path.name for path in (run_dir / "traces").iterdir()) == {
        "D0_seed100_fixed.json",
        "D0_seed101_fixed.json",
        "D0_seed102_fixed.json",
    }
    assert set(path.name for path in run_dir.iterdir()) == {
        "resolved_config.json",
        "metrics.json",
        "summary.png",
        "report.md",
        "traces",
    }
    assert training_report.read_text() == "training report\n"
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_closed_loop(
            run_dir=run_dir,
            resolved=resolved,
            rows=rows,
            force_samples=samples,
            trace_payloads=traces,
            selected_trace_keys={rollout_key("D0", 100, "fixed")},
        )


def test_successful_evaluation_omits_traces(tmp_path) -> None:
    source_run, source_resolved = _source_run(tmp_path)
    run_dir, resolved = prepare_evaluation_run(
        source_checkpoint=source_run / "checkpoints" / "best.pt",
        requested_protocol=source_resolved["evaluation_protocol"],
        policy="act",
    )
    result = _result(success=True, forces=[10.0])
    row, samples = _row("D0", 100, "fixed", result)
    publish_closed_loop(
        run_dir=run_dir,
        resolved=resolved,
        rows=[row],
        force_samples={row["rollout_key"]: samples},
    )
    assert set(path.name for path in run_dir.iterdir()) == {
        "resolved_config.json",
        "metrics.json",
        "summary.png",
        "report.md",
    }
