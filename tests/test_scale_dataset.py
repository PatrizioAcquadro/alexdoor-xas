"""Synthetic regressions for resumable scale-master selection/publication."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from alexdoor_xas.calibration.alex_v2_door import calibration_fingerprint
from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.recording import write_episode
from conftest import FakeForceDoorPushEnv

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scale_module():
    path = REPO_ROOT / "scripts" / "build_scale_dataset.py"
    spec = importlib.util.spec_from_file_location("build_scale_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_scale_plan_is_strict_randomized_and_seed_disjoint(tmp_path) -> None:
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    source = REPO_ROOT / "configs/door_pose_plan_v3_scale.json"
    plan = scale._load_plan(source, config)
    assert plan["fixed_candidates_per_pose"] == 0
    assert plan["randomized_candidates_only"] is True
    assert plan["source_candidates_per_pose"] == 110
    assert plan["selected_episodes_per_pose"] == 110
    assert plan["overdraw_candidates_per_pose"] > 0

    drifted = json.loads(source.read_text())
    drifted["poses"][1]["overdraw_seed_start"] = drifted["poses"][0][
        "source_seed_start"
    ]
    drifted["poses"][1]["overdraw_seed_stop"] = (
        drifted["poses"][1]["overdraw_seed_start"]
        + drifted["overdraw_candidates_per_pose"]
    )
    bad = tmp_path / "overlap.json"
    bad.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="overlap|seed ranges"):
        scale._load_plan(bad, config)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda plan: plan.__setitem__("calibration_fingerprint", "0" * 64), "calibration"),
        (lambda plan: plan["poses"][1].__setitem__("door_yaw_rad", 0.125), "geometry"),
        (lambda plan: plan["poses"][2].__setitem__("source_seed_start", 30_001), "seed"),
    ],
)
def test_scale_plan_binds_canonical_calibration_geometry_and_exact_seed_ranges(
    tmp_path, mutation, message
) -> None:
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    payload = json.loads((REPO_ROOT / "configs/door_pose_plan_v3_scale.json").read_text())
    mutation(payload)
    bad = tmp_path / "bad-plan.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        scale._load_plan(bad, config)


def _candidate_fixture(tmp_path: Path):
    run = tmp_path / "poseD0_attempt001"
    episodes = run / "episodes"
    for seed in (10, 11, 20, 21):
        episode = run_episode(
            FakeForceDoorPushEnv(
                start_door_frame=(0.7, 0.2 + 0.001 * seed, 0.0)
            ),
            plan_episodes(0, 1, seed)[0],
            DataEngineCfg(task="door_push", door_pose_id="D0"),
        )
        if seed == 10:
            assert episode.outcome is not None
            episode.outcome = replace(
                episode.outcome,
                success=False,
                failure_label="synthetic_source_failure",
            )
        write_episode(episode, episodes)
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "sanity.json").write_text(
        json.dumps({"n_episodes_checked": 4}) + "\n"
    )
    plan = {
        "selected_episodes_per_pose": 2,
        "poses": [
            {
                "pose_id": "D0",
                "source_seed_start": 10,
                "source_seed_stop": 12,
                "overdraw_seed_start": 20,
                "overdraw_seed_stop": 22,
            }
        ],
    }
    state = {"poses": {"D0": {"completed": str(run)}}}
    return plan, state


def test_master_selection_uses_overdraw_only_for_failed_source_and_records_provenance(
    tmp_path,
) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    assert [episode.meta.seed for episode in selected] == [11, 20]
    assert len(paths) == 2
    rows = {row["seed"]: row for row in provenance}
    assert rows[10]["decision"] == "SKIPPED"
    assert rows[10]["namespace"] == "source"
    assert any("task failure" in reason for reason in rows[10]["reasons"])
    assert rows[11]["decision"] == "SELECTED"
    assert rows[20]["decision"] == "SELECTED"
    assert rows[20]["namespace"] == "overdraw"
    assert rows[20]["replacement_for_seed"] == 10
    assert rows[21]["decision"] == "NOT_NEEDED_OVERDRAW"
    assert rows[21]["replacement_for_seed"] is None
    assert all(episode.extras["variation"] is not None for episode in selected)


def test_master_selection_fails_closed_when_overdraw_cannot_fill_quota(tmp_path) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    plan["selected_episodes_per_pose"] = 4
    with pytest.raises(RuntimeError, match="need 4"):
        scale._select_master(plan, state)


def test_candidate_ledger_rejects_deleted_row_and_false_replacement_link(tmp_path) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    selected_ids = [episode.meta.episode_id for episode in selected]

    validator = scale._validate_candidate_provenance
    assert validator(
        plan,
        provenance,
        selected_episode_ids=selected_ids,
        expected_source_fingerprint=scale._source_fingerprint(paths),
        require_source_paths=True,
        candidate_state=state,
    )["status"] == "PASS"

    with pytest.raises(ValueError, match="inventory"):
        validator(
            plan,
            provenance[:-1],
            selected_episode_ids=selected_ids,
            expected_source_fingerprint=scale._source_fingerprint(paths),
            require_source_paths=True,
            candidate_state=state,
        )

    falsified = json.loads(json.dumps(provenance))
    replacement = next(row for row in falsified if row["seed"] == 20)
    replacement["replacement_for_seed"] = 999
    with pytest.raises(ValueError, match="replacement"):
        validator(
            plan,
            falsified,
            selected_episode_ids=selected_ids,
            expected_source_fingerprint=scale._source_fingerprint(paths),
            require_source_paths=True,
            candidate_state=state,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda row, tmp: row.__setitem__("episode_id", "false-episode-id"), "episode"),
        (
            lambda row, tmp: row.__setitem__("source_path", str(tmp / "missing.hdf5")),
            "source path|missing",
        ),
        (
            lambda row, tmp: row.__setitem__("source_path", str(tmp.parent / "escape.hdf5")),
            "source path|evidence|missing|escape",
        ),
        (lambda row, tmp: row.__setitem__("content_group_sha256", "f" * 64), "content"),
    ],
)
def test_unused_overdraw_rows_are_authenticated_against_raw_candidates(
    tmp_path, mutation, message
) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    tampered = json.loads(json.dumps(provenance))
    unused = next(row for row in tampered if row["decision"] == "NOT_NEEDED_OVERDRAW")
    mutation(unused, tmp_path)

    with pytest.raises(ValueError, match=message):
        scale._validate_candidate_provenance(
            plan,
            tampered,
            selected_episode_ids=[episode.meta.episode_id for episode in selected],
            expected_source_fingerprint=scale._source_fingerprint(paths),
            require_source_paths=True,
            candidate_state=state,
        )


def test_candidate_replay_rejects_invented_unused_decision_and_skipped_reason(
    tmp_path,
) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    selected_ids = [episode.meta.episode_id for episode in selected]
    source_fingerprint = scale._source_fingerprint(paths)

    invented = json.loads(json.dumps(provenance))
    unused = next(row for row in invented if row["decision"] == "NOT_NEEDED_OVERDRAW")
    unused["decision"] = "SKIPPED"
    unused["reasons"] = ["invented rejection"]
    with pytest.raises(ValueError, match="decision|reason|replay"):
        scale._validate_candidate_provenance(
            plan,
            invented,
            selected_episode_ids=selected_ids,
            expected_source_fingerprint=source_fingerprint,
            require_source_paths=True,
            candidate_state=state,
        )

    wrong_reason = json.loads(json.dumps(provenance))
    skipped = next(row for row in wrong_reason if row["decision"] == "SKIPPED")
    skipped["reasons"] = ["invented rejection"]
    with pytest.raises(ValueError, match="reason|replay"):
        scale._validate_candidate_provenance(
            plan,
            wrong_reason,
            selected_episode_ids=selected_ids,
            expected_source_fingerprint=source_fingerprint,
            require_source_paths=True,
            candidate_state=state,
        )


def test_candidate_raw_replay_defeats_refreshed_ledger_and_report_hashes(tmp_path) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    tampered = json.loads(json.dumps(provenance))
    unused = next(row for row in tampered if row["decision"] == "NOT_NEEDED_OVERDRAW")
    unused["episode_id"] = "forged-unused-candidate"
    refreshed_ledger_hash = scale.hashlib.sha256(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    refreshed_report_hash = scale.hashlib.sha256(
        json.dumps(
            {"candidate_provenance_sha256": refreshed_ledger_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert len(refreshed_ledger_hash) == len(refreshed_report_hash) == 64

    with pytest.raises(ValueError, match="episode|replay"):
        scale._validate_candidate_provenance(
            plan,
            tampered,
            selected_episode_ids=[episode.meta.episode_id for episode in selected],
            expected_source_fingerprint=scale._source_fingerprint(paths),
            require_source_paths=True,
            candidate_state=state,
        )


def test_candidate_replay_rejects_nondeterministic_row_and_replacement_order(
    tmp_path,
) -> None:
    scale = _scale_module()
    plan, state = _candidate_fixture(tmp_path)
    selected, paths, provenance = scale._select_master(plan, state)
    selected_ids = [episode.meta.episode_id for episode in selected]
    source_fingerprint = scale._source_fingerprint(paths)

    reordered = json.loads(json.dumps(provenance))
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="order|replay"):
        scale._validate_candidate_provenance(
            plan,
            reordered,
            selected_episode_ids=selected_ids,
            expected_source_fingerprint=source_fingerprint,
            require_source_paths=True,
            candidate_state=state,
        )

    second_plan, second_state = _candidate_fixture(tmp_path / "second")
    episodes_dir = Path(second_state["poses"]["D0"]["completed"]) / "episodes"
    for seed in (11,):
        episode_path = next(
            path for path in episodes_dir.glob("episode_*.hdf5")
            if scale.read_episode(path).meta.seed == seed
        )
        episode = scale.read_episode(episode_path)
        assert episode.outcome is not None
        episode.outcome = replace(
            episode.outcome,
            success=False,
            failure_label="second_synthetic_source_failure",
        )
        episode_path.unlink()
        episode_path.with_suffix(".meta.json").unlink()
        write_episode(episode, episodes_dir)
    selected, paths, provenance = scale._select_master(second_plan, second_state)
    swapped = json.loads(json.dumps(provenance))
    replacements = [
        row for row in swapped
        if row["decision"] == "SELECTED" and row["namespace"] == "overdraw"
    ]
    assert len(replacements) == 2
    replacements[0]["replacement_for_seed"], replacements[1]["replacement_for_seed"] = (
        replacements[1]["replacement_for_seed"],
        replacements[0]["replacement_for_seed"],
    )
    with pytest.raises(ValueError, match="replacement|order|replay"):
        scale._validate_candidate_provenance(
            second_plan,
            swapped,
            selected_episode_ids=[episode.meta.episode_id for episode in selected],
            expected_source_fingerprint=scale._source_fingerprint(paths),
            require_source_paths=True,
            candidate_state=second_state,
        )


def test_real_candidate_ledger_replays_all_750_raw_candidates() -> None:
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    plan = scale._load_plan(REPO_ROOT / config.selection.pose_plan, config)
    master = json.loads((REPO_ROOT / config.dataset.master_manifest).read_text())
    state = json.loads(
        (REPO_ROOT / "outputs/v3_scale_generation/generation_state.json").read_text()
    )
    report = scale._validate_candidate_provenance(
        plan,
        master["candidate_provenance"],
        selected_episode_ids=master["selected_episode_ids"],
        expected_source_fingerprint=master["source_fingerprint_sha256"],
        require_source_paths=True,
        candidate_state=state,
    )
    assert report["candidate_count"] == 750
    assert report["selected_count"] == 550
    assert report["decision_counts"] == {
        "NOT_NEEDED_OVERDRAW": 200,
        "SELECTED": 550,
        "SKIPPED": 0,
    }
    assert report["candidate_provenance_sha256"] == (
        "41991acbe90a3a559720e02b7a34d71c2ccf90c4c0e2fb45c26f37eeb97b0000"
    )
    assert report["source_fingerprint_sha256"] == (
        "79dd3e819c2fbb2d21b9cf3848df7942bd6e69f163a9d82ef892710f5e39d27b"
    )


def _isolated_scale_contract(tmp_path: Path, monkeypatch):
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    for relative in (
        config.selection.pose_plan,
        config.selection.canonical_pose_plan,
        config.selection.calibration,
        config.dataset.master_manifest,
    ):
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    monkeypatch.setattr(scale.paths, "REPO_ROOT", tmp_path)
    return scale, config, tmp_path / config.selection.pose_plan


def test_scale_plan_recomputes_calibration_self_fingerprint(tmp_path, monkeypatch) -> None:
    scale, config, plan_path = _isolated_scale_contract(tmp_path, monkeypatch)
    calibration_path = tmp_path / config.selection.calibration
    calibration = json.loads(calibration_path.read_text())
    calibration["reach_shell_m"][1] = 0.79
    calibration_path.write_text(json.dumps(calibration))

    with pytest.raises(ValueError, match="calibration.*fingerprint|fingerprint.*calibration"):
        scale._load_plan(plan_path, config)


def test_scale_plan_keeps_canonical_fingerprint_when_calibration_is_refreshed(
    tmp_path, monkeypatch
) -> None:
    scale, config, plan_path = _isolated_scale_contract(tmp_path, monkeypatch)
    calibration_path = tmp_path / config.selection.calibration
    calibration = json.loads(calibration_path.read_text())
    calibration["reach_shell_m"][1] = 0.79
    calibration["fingerprint"] = calibration_fingerprint(calibration)
    calibration_path.write_text(json.dumps(calibration))

    with pytest.raises(ValueError, match="calibration.*fingerprint|fingerprint.*calibration"):
        scale._load_plan(plan_path, config)


def test_scale_plan_rejects_jointly_refreshed_noncanonical_calibration(
    tmp_path, monkeypatch
) -> None:
    scale, config, plan_path = _isolated_scale_contract(tmp_path, monkeypatch)
    calibration_path = tmp_path / config.selection.calibration
    calibration = json.loads(calibration_path.read_text())
    calibration["robot_asset"]["id"] = "different-runtime-asset:" + "d" * 64
    calibration["robot_asset"]["sha256"] = "d" * 64
    calibration["robot_asset"]["manifest_fingerprint"] = "d" * 64
    calibration["fingerprint"] = calibration_fingerprint(calibration)
    calibration_path.write_text(json.dumps(calibration))
    plan = json.loads(plan_path.read_text())
    plan["calibration_fingerprint"] = calibration["fingerprint"]
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="calibration|robot asset|master"):
        scale._load_plan(plan_path, config)


def test_scale_plan_rejects_jointly_refreshed_runtime_contract(
    tmp_path, monkeypatch
) -> None:
    scale, config, plan_path = _isolated_scale_contract(tmp_path, monkeypatch)
    calibration_path = tmp_path / config.selection.calibration
    calibration = json.loads(calibration_path.read_text())
    calibration["runtime_versions"]["isaac_lab"] = "3.1.0"
    calibration["fingerprint"] = calibration_fingerprint(calibration)
    calibration_path.write_text(json.dumps(calibration))
    plan = json.loads(plan_path.read_text())
    plan["calibration_fingerprint"] = calibration["fingerprint"]
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="calibration|runtime"):
        scale._load_plan(plan_path, config)


def test_canonical_scale_calibration_passes_shared_validated_loader() -> None:
    scale = _scale_module()
    config = load_sweep_config(REPO_ROOT / "configs/cluster_sweep.v1.json")
    plan = scale._load_plan(REPO_ROOT / config.selection.pose_plan, config)
    assert plan["calibration_fingerprint"] == (
        "066e0a2d0157549a331b96702e643fd7626eed58d33bbe701005e111ed358948"
    )
