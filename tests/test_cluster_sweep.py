"""Regression contract for the full nested Gilbreth sweep package."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.cluster_sweep.config import SweepConfigError, load_sweep_config
from alexdoor_xas.cluster_sweep.preflight import ClusterPreflightError, run_sweep_preflight
from alexdoor_xas.cluster_sweep.returns import (
    SweepReturnError,
    build_sweep_return_manifest,
    verify_sweep_checkpoints,
    verify_sweep_return_manifest,
)
from alexdoor_xas.cluster_sweep.slurm import render_sweep_slurm_script
from alexdoor_xas.cluster_sweep.transfer import (
    SweepTransferError,
    build_sweep_transfer_manifest,
    secret_problems,
    verify_sweep_transfer_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cluster_sweep.v1.json"


@pytest.fixture(scope="module")
def config():
    return load_sweep_config(CONFIG_PATH)


def test_sweep_config_freezes_master_views_and_16_stable_cells(config) -> None:
    assert config.schema == "alexdoor_xas.cluster_sweep_config.v1"
    assert config.dataset.task == "door_push_alex_v2"
    assert config.dataset.master_version == "v3_scale_master"
    assert config.dataset.obs_preset == "core_door_pose"
    assert config.dataset.pose_ids == ("D0", "D1", "D2", "D3", "D4")
    assert config.dataset.master_count == 550
    assert config.dataset.episodes_per_pose == 110
    assert [(view.view_id, view.train, view.val, view.test) for view in config.views] == [
        ("v3_scale_n50", 50, 25, 25),
        ("v3_scale_n100", 100, 25, 25),
        ("v3_scale_n250", 250, 25, 25),
        ("v3_scale_n500", 500, 25, 25),
    ]
    assert config.training.seed == 0
    assert config.training.device == "cuda"
    assert config.training.overfit_episodes is None
    assert config.training.wandb_mode == "offline"
    assert config.training.distributed is False
    assert len(config.cells) == 16
    assert [cell.index for cell in config.cells] == list(range(16))
    assert len({cell.run_id for cell in config.cells}) == 16
    assert config.cells[0].run_id == "sweep_act_a2_n50_seed0"
    assert config.cells[1].run_id == "sweep_act_a3_n50_seed0"
    assert config.cells[2].run_id == "sweep_diffusion_a2_n50_seed0"
    assert config.cells[3].run_id == "sweep_diffusion_a3_n50_seed0"
    assert config.cells[-1].run_id == "sweep_diffusion_a3_n500_seed0"
    assert config.slurm.gpus_per_task == 1
    assert config.slurm.array_max_concurrent == 2
    assert config.environment.require_no_isaac is True


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("dataset", "master_count"), 549, "550"),
        (("dataset", "episodes_per_pose"), 109, "110"),
        (("views", 0, "train"), 49, "view"),
        (("training", "overfit_episodes"), 2, "overfit"),
        (("training", "wandb_mode"), "online", "offline"),
        (("training", "distributed"), True, "distributed"),
        (("slurm", "gpus_per_task"), 2, "one GPU"),
    ],
)
def test_sweep_config_rejects_contract_drift(tmp_path, path, value, message) -> None:
    payload = json.loads(CONFIG_PATH.read_text())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(SweepConfigError, match=message):
        load_sweep_config(bad)


def test_sweep_config_rejects_unknown_keys_duplicate_cells_and_run_id_drift(
    tmp_path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["unexpected"] = True
    bad = tmp_path / "unknown.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(SweepConfigError, match="extra"):
        load_sweep_config(bad)

    payload = json.loads(CONFIG_PATH.read_text())
    payload["cells"][1] = copy.deepcopy(payload["cells"][0])
    bad = tmp_path / "duplicate.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(SweepConfigError, match="cell|duplicate|mapping"):
        load_sweep_config(bad)

    payload = json.loads(CONFIG_PATH.read_text())
    payload["cells"][0]["run_id"] = "drifted"
    bad = tmp_path / "run-id.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(SweepConfigError, match="run_id|mapping"):
        load_sweep_config(bad)


def test_training_cells_preserve_committed_non_pilot_defaults(config) -> None:
    for cell in config.cells:
        assert cell.space in {"A2_ee_delta", "A3_obj_rel_ee_delta"}
        assert cell.view_id in {
            "v3_scale_n50",
            "v3_scale_n100",
            "v3_scale_n250",
            "v3_scale_n500",
        }
        assert all("deepspeed" not in key.lower() for key in cell.overrides)
        assert all("ddp" not in key.lower() for key in cell.overrides)
        if cell.policy == "act":
            assert cell.entrypoint == "scripts/train_act.py"
            assert cell.overrides == {"train.epochs": 100, "train.val_every": 5}
        else:
            assert cell.entrypoint == "scripts/train_diffusion.py"
            assert cell.overrides == {
                "train.epochs": 300,
                "train.val_every": 10,
                "train.use_ema": True,
                "train.val_inference_steps": 10,
            }


def test_sweep_slurm_is_16_cell_single_gpu_non_isaac_and_prefix_python(config) -> None:
    rendered = render_sweep_slurm_script(
        config,
        source_commit="1" * 40,
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/results"),
        account="example-account",
        partition="example-partition",
        qos=None,
    )
    assert "#SBATCH --array=0-15%2" in rendered
    assert "#SBATCH --gpus-per-node=1" in rendered
    assert "dataset.version=v3_scale_master" in rendered
    assert "dataset.view_id=$VIEW_ID" in rendered
    assert "dataset.obs_preset=core_door_pose" in rendered
    assert "train.overfit_episodes=null" in rendered
    assert "+wandb.mode=offline" in rendered
    assert '"$CONDA_PREFIX/bin/python"' in rendered
    assert "conda activate" not in rendered.lower()
    assert "bin/activate" not in rendered
    assert "isaaclab" not in rendered.lower()
    assert "deepspeed" not in rendered.lower()
    assert "torchrun" not in rendered.lower()
    assert 'rm -rf "$publish_tmp"' not in rendered
    assert "completion.json" in rendered
    assert "failure.json" in rendered
    assert "publication_report.json" in rendered
    assert 'ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"' in rendered
    assert 'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"' in rendered
    assert rendered.count("RUN_ID=sweep_") == 16

    result = subprocess.run(
        ["bash", "-n"], input=rendered, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_sweep_slurm_concurrency_and_resources_are_render_time_configurable(config) -> None:
    selected = replace(config, slurm=replace(config.slurm, array_max_concurrent=7))
    rendered = render_sweep_slurm_script(
        selected,
        source_commit="2" * 40,
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/results"),
        account="acct",
        partition="gpu",
        qos="normal",
        memory="96G",
        cpus_per_task=16,
        wall_time="24:00:00",
    )
    assert "#SBATCH --array=0-15%7" in rendered
    assert "#SBATCH --mem=96G" in rendered
    assert "#SBATCH --cpus-per-task=16" in rendered
    assert "#SBATCH --time=24:00:00" in rendered
    assert "#SBATCH --qos=normal" in rendered


def test_sweep_sources_have_no_isaac_imports_or_secret_material(config) -> None:
    for relative in config.tracked_transfer_files:
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        content = path.read_bytes()
        assert secret_problems(relative, content) == []
        if relative.startswith("src/alexdoor_xas/cluster_sweep/"):
            lowered = content.lower()
            assert b"import isaac" not in lowered
            assert b"from isaac" not in lowered


def _source_state(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "clean_tree": clean,
        "commit_time": "2026-07-13T12:00:00+00:00",
    }


def test_transfer_manifest_is_exact_hash_bound_and_clean_tree_gated(
    tmp_path, config, monkeypatch
) -> None:
    import alexdoor_xas.cluster_sweep.transfer as transfer

    first = tmp_path / "first.json"
    second = tmp_path / "second.bin"
    first.write_text('{"safe": true}\n')
    second.write_bytes(b"payload")
    contract = {"master_version": config.dataset.master_version}
    asset = {"sha256": "a" * 64, "transferred": False}
    monkeypatch.setattr(
        transfer,
        "_collect_contract",
        lambda root, selected, require_tracked: (
            [(first, "metadata"), (second, "dataset")],
            contract,
            asset,
        ),
    )
    manifest = build_sweep_transfer_manifest(
        tmp_path, config, source_state=_source_state()
    )
    assert verify_sweep_transfer_manifest(
        manifest, tmp_path, config, source_state=_source_state()
    ) == []
    second.write_bytes(b"tampered")
    failures = verify_sweep_transfer_manifest(
        manifest, tmp_path, config, source_state=_source_state()
    )
    assert any("hash mismatch" in failure for failure in failures)
    with pytest.raises(SweepTransferError, match="clean"):
        build_sweep_transfer_manifest(
            tmp_path, config, source_state=_source_state(clean=False)
        )


def _transfer_manifest(config) -> dict[str, object]:
    splits = {"train": ["train"], "val": ["val"], "test": ["test"]}
    views = {
        view.view_id: {
            "view_fingerprint_sha256": str(index + 1) * 64,
            "splits": splits,
        }
        for index, view in enumerate(config.views)
    }
    norms = {
        f"{cell.space}:{cell.view_id}": {
            "sha256": "a" * 64,
            "normalization_fingerprint_sha256": "b" * 64,
        }
        for cell in config.cells
    }
    spaces = {
        space: {"dataset_fingerprint_sha256": "c" * 64}
        for space in config.dataset.spaces
    }
    return {
        "sweep_id": config.sweep_id,
        "source_git": {"commit": "1" * 40},
        "dataset": {
            "views": views,
            "normalization_artifacts": norms,
            "spaces": spaces,
        },
    }


def _write_complete_attempt(root: Path, config, attempt_id: str = "12345") -> None:
    required = (
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "logs/train_log.json",
        "metrics/open_loop.json",
        "resolved_config.json",
        "environment/environment_inventory.json",
        "environment/requirements.lock",
        "environment/preflight_report.json",
        "slurm/stdout.log",
        "slurm/stderr.log",
    )
    for cell in config.cells:
        run = root / "attempts" / attempt_id / str(cell.index) / cell.run_id
        for relative in required:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        publication = run / "wandb" / "publication_report.json"
        publication.parent.mkdir(parents=True)
        publication.write_text(
            json.dumps(
                {
                    "destination_contains_symlinks": False,
                    "destination_symlink_count": 0,
                }
            )
        )
        completion = run / "status" / "completion.json"
        completion.parent.mkdir(parents=True)
        completion.write_text(
            json.dumps(
                {
                    "schema": "alexdoor_xas.cluster_sweep_cell_status.v1",
                    "status": "COMPLETED",
                    "run_id": cell.run_id,
                    "policy": cell.policy,
                    "space": cell.space,
                    "view_id": cell.view_id,
                    "exit_code": 0,
                    "source_git_commit": "1" * 40,
                    "attempt": {
                        "slurm_array_job_id": attempt_id,
                        "slurm_array_task_id": str(cell.index),
                        "run_id": cell.run_id,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_return_manifest_requires_one_complete_exact_16_cell_attempt(
    tmp_path, config
) -> None:
    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)
    manifest = build_sweep_return_manifest(
        tmp_path, attempt_id="12345", config=config, transfer_manifest=transfer
    )
    assert manifest["cell_count"] == 16
    assert len(manifest["cells"]) == 16
    assert verify_sweep_return_manifest(
        manifest,
        tmp_path,
        attempt_id="12345",
        config=config,
        transfer_manifest=transfer,
    ) == []

    target = tmp_path / manifest["files"][0]["path"]
    target.write_bytes(target.read_bytes() + b"tamper")
    failures = verify_sweep_return_manifest(
        manifest,
        tmp_path,
        attempt_id="12345",
        config=config,
        transfer_manifest=transfer,
    )
    assert any("hash mismatch" in failure for failure in failures)


def test_return_rejects_failed_missing_and_mixed_attempt_artifacts(tmp_path, config) -> None:
    transfer = _transfer_manifest(config)
    _write_complete_attempt(tmp_path, config)
    first = config.cells[0]
    failure = (
        tmp_path
        / "attempts/12345"
        / str(first.index)
        / first.run_id
        / "status/failure.json"
    )
    failure.write_text("{}\n")
    with pytest.raises(SweepReturnError, match="partial|failed"):
        build_sweep_return_manifest(
            tmp_path, attempt_id="12345", config=config, transfer_manifest=transfer
        )
    failure.unlink()
    task = tmp_path / "attempts/12345/15"
    moved = tmp_path / "attempts/12345/mixed"
    task.rename(moved)
    with pytest.raises(SweepReturnError, match="inventory"):
        build_sweep_return_manifest(
            tmp_path, attempt_id="12345", config=config, transfer_manifest=transfer
        )


def test_return_checkpoint_cpu_load_verifies_all_cell_provenance(tmp_path, config) -> None:
    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)

    def loader(path: Path):
        run_id = path.parents[1].name
        cell = next(item for item in config.cells if item.run_id == run_id)
        splits = transfer["dataset"]["views"][cell.view_id]["splits"]
        split_digest = __import__("hashlib").sha256(
            json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return SimpleNamespace(
            config={
                "dataset": {
                    "task": config.dataset.task,
                    "space": cell.space,
                    "version": config.dataset.master_version,
                    "view_id": cell.view_id,
                    "obs_preset": config.dataset.obs_preset,
                }
            },
            provenance={
                "master_dataset_fingerprint_sha256": "c" * 64,
                "view_id": cell.view_id,
                "view_fingerprint_sha256": transfer["dataset"]["views"][cell.view_id][
                    "view_fingerprint_sha256"
                ],
                "split_episode_ids": splits,
                "split_counts": {"train": 1, "val": 1, "test": 1},
                "split_fingerprint_sha256": split_digest,
                "normalization_sha256": "a" * 64,
                "normalization_fingerprint_sha256": "b" * 64,
                "source_git_commit": "1" * 40,
                "action_space": cell.space,
                "obs_preset": config.dataset.obs_preset,
            },
        )

    statuses = verify_sweep_checkpoints(
        tmp_path,
        attempt_id="12345",
        config=config,
        transfer_manifest=transfer,
        loaders={"act": loader, "diffusion": loader},
    )
    assert statuses == {cell.run_id: "CPU_LOAD_PASS" for cell in config.cells}


def test_sweep_preflight_is_non_isaac_offline_and_dependency_complete(
    tmp_path, config, monkeypatch
) -> None:
    import alexdoor_xas.cluster_sweep.preflight as preflight
    from alexdoor_xas.cluster_pilot.preflight import REQUIRED_IMPORTS

    manifest = {
        "schema": "alexdoor_xas.cluster_sweep_transfer_manifest.v1",
        "source_git": {"commit": "1" * 40},
    }
    dependencies = {"python": "3.11.9", "ruff": "ruff 1.0"}
    dependencies.update({name: "1.0" for name in REQUIRED_IMPORTS})
    monkeypatch.setattr(preflight, "verify_sweep_transfer_manifest", lambda *a, **k: [])

    def checkpoint(path: Path) -> None:
        path.write_bytes(b"checkpoint")

    report = run_sweep_preflight(
        repo_root=tmp_path,
        config=config,
        manifest=manifest,
        scratch_output=tmp_path,
        source_state=_source_state(),
        dependency_probe=lambda: dependencies,
        module_probe=lambda: {},
        checkpoint_probe=checkpoint,
    )
    assert report["status"] == "PASS"
    assert report["wandb_mode"] == "offline"
    with pytest.raises(ClusterPreflightError, match="simulator"):
        run_sweep_preflight(
            repo_root=tmp_path,
            config=config,
            manifest=manifest,
            scratch_output=tmp_path,
            source_state=_source_state(),
            dependency_probe=lambda: dependencies,
            module_probe=lambda: {"omni": "AVAILABLE"},
            checkpoint_probe=checkpoint,
        )
    with pytest.raises(ClusterPreflightError, match="incomplete"):
        run_sweep_preflight(
            repo_root=tmp_path,
            config=config,
            manifest=manifest,
            scratch_output=tmp_path,
            source_state=_source_state(),
            dependency_probe=lambda: {"python": "3.11.9"},
            module_probe=lambda: {},
            checkpoint_probe=checkpoint,
        )
