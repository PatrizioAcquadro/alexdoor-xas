"""Regression contract for the full nested Gilbreth sweep package."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
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
    write_return_artifacts,
)
from alexdoor_xas.cluster_sweep.slurm import render_sweep_slurm_script
from alexdoor_xas.cluster_sweep.transfer import (
    SweepTransferError,
    build_sweep_transfer_manifest,
    secret_problems,
    verify_sweep_transfer_manifest,
)
from alexdoor_xas.policies.act import load_act_config
from alexdoor_xas.policies.diffusion import load_diffusion_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cluster_sweep.v1.json"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


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
    assert config.storage.conda_prefix_relative == "envs/alexdoor-gilbreth-pilot-py311"
    assert config.environment.numpy_version == "2.4.6"
    assert config.environment.torch_version == "2.12.1+cu126"
    assert config.environment.torch_cuda_version == "12.6"
    assert config.selection.pose_plan == "configs/door_pose_plan_v3_scale.json"
    assert config.selection.canonical_pose_plan == "configs/door_pose_plan_v2_pose.json"
    assert config.selection.calibration == "configs/alex_v2_door_calibration.v0.json"


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
        (("storage", "conda_prefix_relative"), "envs/unsupported", "pilot"),
        (("environment", "numpy_version"), "2.4.5", "NumPy"),
        (("environment", "torch_version"), "2.12.0", "PyTorch"),
        (("environment", "torch_cuda_version"), "12.5", "CUDA"),
        (("selection", "pose_plan"), "configs/drifted.json", "pose plan"),
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


def test_rendered_sweep_slurm_executes_with_polluted_path_and_isolates_attempts(
    tmp_path, config
) -> None:
    depot = tmp_path / "depot"
    scratch = tmp_path / "scratch"
    durable = depot / "durable"
    repo = depot / config.storage.source_checkout_relative
    prefix = depot / config.storage.conda_prefix_relative
    polluted = tmp_path / "polluted"
    commit = "1" * 40
    (repo / ".git").mkdir(parents=True)
    manifest = repo / "outputs/cluster_sweep/sweep_transfer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    scratch.mkdir()
    calls = tmp_path / "prefix-python-calls.txt"
    polluted_marker = tmp_path / "polluted-python-ran.txt"

    _write_executable(
        polluted / "git",
        f"""#!/bin/sh
case "$1:$2" in
  rev-parse:HEAD) echo {commit}; exit 0 ;;
  status:--porcelain) exit 0 ;;
  *) exit 98 ;;
esac
""",
    )
    _write_executable(
        polluted / "python",
        "#!/bin/sh\necho ran > \"$POLLUTED_PYTHON_MARKER\"\nexit 97\n",
    )
    _write_executable(
        prefix / "bin/python",
        """#!/bin/bash
set -eu
printf '%s|%s\n' "$0" "$*" >> "$PREFIX_CALL_LOG"
case "$1" in
  scripts/build_cluster_sweep_manifest.py)
    exit 0
    ;;
  scripts/preflight_cluster_sweep.py)
    report=""; environment_dir=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --report) report=$2; shift 2 ;;
        --environment-dir) environment_dir=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$environment_dir" "$(dirname "$report")"
    printf '{}\n' > "$environment_dir/environment_inventory.json"
    printf 'synthetic==1\n' > "$environment_dir/requirements.lock"
    printf '{}\n' > "$report"
    [[ "${FAKE_FAIL_STAGE:-}" != preflight ]] || exit 41
    exit 0
    ;;
  scripts/train_act.py|scripts/train_diffusion.py)
    entry=$1; shift
    output_root=""; run_id=""; wandb_dir=""
    for arg in "$@"; do
      case "$arg" in
        run.output_root=*) output_root=${arg#*=} ;;
        run.run_id=*) run_id=${arg#*=} ;;
        +wandb.dir=*) wandb_dir=${arg#*=} ;;
      esac
    done
    [[ "${FAKE_FAIL_STAGE:-}" != training ]] || exit 42
    if [[ "$entry" == scripts/train_act.py ]]; then
      experiment=act_door_push
    else
      experiment=diffusion_door_push
    fi
    run_dir="$output_root/$experiment/$run_id"
    mkdir -p "$run_dir/checkpoints" "$run_dir/logs" "$run_dir/metrics" "$wandb_dir/run"
    printf 'best\n' > "$run_dir/checkpoints/best.pt"
    printf 'last\n' > "$run_dir/checkpoints/last.pt"
    printf '{}\n' > "$run_dir/logs/train_log.json"
    printf '{}\n' > "$run_dir/metrics/open_loop.json"
    printf '{}\n' > "$run_dir/resolved_config.json"
    printf 'wandb\n' > "$wandb_dir/run/data.txt"
    ln -s data.txt "$wandb_dir/run/unsafe-link"
    exit 0
    ;;
  src/alexdoor_xas/cluster_pilot/wandb_publication.py)
    source=""; destination=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --source) source=$2; shift 2 ;;
        --destination) destination=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$destination"
    while IFS= read -r -d '' file; do
      cp "$file" "$destination/$(basename "$file")"
    done < <(find "$source" -type f -print0)
    printf '{"destination_contains_symlinks":false,"destination_symlink_count":0}\n' \
      > "$destination/publication_report.json"
    exit 0
    ;;
  *) exit 99 ;;
esac
""",
    )
    rendered = render_sweep_slurm_script(
        config,
        source_commit=commit,
        depot_root=depot,
        scratch_root=scratch,
        durable_results_root=durable,
        account="acct",
        partition="gpu",
        qos=None,
    )
    script = tmp_path / "sweep.slurm"
    _write_executable(script, rendered)
    assert subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    ).returncode == 0

    base_env = {
        **os.environ,
        "PATH": f"{polluted}:/usr/bin:/bin",
        "PREFIX_CALL_LOG": str(calls),
        "POLLUTED_PYTHON_MARKER": str(polluted_marker),
        "SLURM_ARRAY_TASK_ID": "0",
    }

    def run(attempt: str, fail_stage: str = ""):
        env = {
            **base_env,
            "SLURM_ARRAY_JOB_ID": attempt,
            "SLURM_JOB_ID": f"{attempt}_0",
            "FAKE_FAIL_STAGE": fail_stage,
        }
        return subprocess.run(
            ["/bin/bash", str(script)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    success = run("500")
    assert success.returncode == 0, success.stderr
    first = durable / "attempts/500/0" / config.cells[0].run_id
    first_status = (first / "status/completion.json").read_bytes()
    assert not any(path.is_symlink() for path in first.rglob("*"))
    assert json.loads((first / "wandb/publication_report.json").read_text())[
        "destination_symlink_count"
    ] == 0

    shutil.rmtree(scratch / config.storage.scratch_runs_relative / "attempts/500")
    duplicate = run("500")
    assert duplicate.returncode == 28
    assert (first / "status/completion.json").read_bytes() == first_status

    retry = run("501")
    assert retry.returncode == 0, retry.stderr
    assert (durable / "attempts/501/0" / config.cells[0].run_id).is_dir()
    preflight_failure = run("502", "preflight")
    assert preflight_failure.returncode == 41
    training_failure = run("503", "training")
    assert training_failure.returncode == 42
    for attempt, code in (("502", 41), ("503", 42)):
        failure = json.loads(
            (
                durable
                / f"attempts/{attempt}/0"
                / config.cells[0].run_id
                / "status/failure.json"
            ).read_text()
        )
        assert failure["exit_code"] == code

    assert not polluted_marker.exists()
    assert calls.read_text().splitlines()
    assert all(
        line.startswith(str(prefix / "bin/python") + "|")
        for line in calls.read_text().splitlines()
    )


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
        lambda root, selected, **kwargs: (
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


def test_transfer_robot_asset_distinguishes_runtime_fingerprint_from_urdf_hash(
    tmp_path,
) -> None:
    import alexdoor_xas.cluster_sweep.transfer as transfer

    urdf = tmp_path / "alex_v2.urdf"
    urdf.write_bytes(b"<robot name='alex_v2'/>\n")
    urdf_sha = __import__("hashlib").sha256(urdf.read_bytes()).hexdigest()
    runtime_sha = "b" * 64
    asset = {
        "id": f"runtime:{runtime_sha}",
        "sha256": runtime_sha,
        "manifest_fingerprint": runtime_sha,
        "manifest": {"fingerprint": runtime_sha, "urdf_sha256": urdf_sha},
    }
    contract = transfer._robot_asset_contract(asset, urdf)
    assert contract["runtime_asset_fingerprint_sha256"] == runtime_sha
    assert contract["urdf_sha256"] == urdf_sha
    assert contract["transferred"] is False

    urdf.write_bytes(b"drift")
    with pytest.raises(SweepTransferError, match="URDF hash"):
        transfer._robot_asset_contract(asset, urdf)


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
        space: {
            "source_fingerprint_sha256": "d" * 64,
            "dataset_fingerprint_sha256": "c" * 64,
        }
        for space in config.dataset.spaces
    }
    return {
        "sweep_id": config.sweep_id,
        "source_git": {"commit": "1" * 40},
        "dataset": {
            "source_fingerprint_sha256": "d" * 64,
            "views": views,
            "normalization_artifacts": norms,
            "spaces": spaces,
        },
    }


def _resolved_cell_config(
    cell,
    *,
    output_root: str = "/scratch/output",
    wandb_dir: str = "/scratch/wandb",
):
    overrides = [
        "dataset.task=door_push_alex_v2",
        f"dataset.space={cell.space}",
        "dataset.version=v3_scale_master",
        f"dataset.view_id={cell.view_id}",
        "dataset.obs_preset=core_door_pose",
        "train.seed=0",
        "train.device=cuda",
        "train.overfit_episodes=null",
        f"run.run_id={cell.run_id}",
        f"run.output_root={output_root}",
        "+wandb.mode=offline",
        f"+wandb.dir={wandb_dir}",
        *[
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in cell.overrides.items()
        ],
    ]
    selected = (
        load_act_config(overrides)
        if cell.policy == "act"
        else load_diffusion_config(overrides)
    )
    return dataclasses.asdict(selected)


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
            if relative == "resolved_config.json":
                path.write_text(
                    json.dumps(_resolved_cell_config(cell), indent=2, sort_keys=True) + "\n"
                )
            else:
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


def test_written_return_file_list_contains_payload_and_two_controls_exactly_once(
    tmp_path, config
) -> None:
    import alexdoor_xas.cluster_sweep.returns as returns

    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)
    manifest = build_sweep_return_manifest(
        tmp_path, attempt_id="12345", config=config, transfer_manifest=transfer
    )
    manifest_path, files_path, _ = write_return_artifacts(tmp_path, manifest)
    expected_controls = {
        ".sweep_return/attempts/12345/return_manifest.json",
        ".sweep_return/attempts/12345/return-files.txt",
    }
    rows = files_path.read_text().splitlines()
    payload = {entry["path"] for entry in manifest["files"]}
    assert set(rows) == payload | expected_controls
    assert len(rows) == len(set(rows))
    assert manifest_path.relative_to(tmp_path).as_posix() in rows
    assert files_path.relative_to(tmp_path).as_posix() in rows
    assert returns.verify_return_control_files(
        tmp_path,
        manifest,
        manifest_path=manifest_path,
        files_path=files_path,
    ) == []

    files_path.write_text(files_path.read_text() + rows[0] + "\n")
    assert any(
        "duplicate" in failure
        for failure in returns.verify_return_control_files(
            tmp_path,
            manifest,
            manifest_path=manifest_path,
            files_path=files_path,
        )
    )


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
        resolved = _resolved_cell_config(cell)
        return SimpleNamespace(
            config=resolved,
            provenance={
                "master_dataset_fingerprint_sha256": "d" * 64,
                "action_dataset_fingerprint_sha256": "c" * 64,
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
                "resolved_training_config_sha256": hashlib.sha256(
                    json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
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


def _checkpoint_loaders(config, transfer, *, configs=None, provenance_updates=None):
    selected_configs = configs or {}
    selected_updates = provenance_updates or {}

    def loader(path: Path):
        run_id = path.parents[1].name
        cell = next(item for item in config.cells if item.run_id == run_id)
        resolved = copy.deepcopy(selected_configs.get(run_id, _resolved_cell_config(cell)))
        splits = transfer["dataset"]["views"][cell.view_id]["splits"]
        provenance = {
            "master_dataset_fingerprint_sha256": "d" * 64,
            "action_dataset_fingerprint_sha256": "c" * 64,
            "view_id": cell.view_id,
            "view_fingerprint_sha256": transfer["dataset"]["views"][cell.view_id][
                "view_fingerprint_sha256"
            ],
            "split_episode_ids": splits,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "split_fingerprint_sha256": hashlib.sha256(
                json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "normalization_sha256": "a" * 64,
            "normalization_fingerprint_sha256": "b" * 64,
            "source_git_commit": "1" * 40,
            "action_space": cell.space,
            "obs_preset": config.dataset.obs_preset,
            "resolved_training_config_sha256": hashlib.sha256(
                json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        provenance.update(selected_updates.get(run_id, {}))
        return SimpleNamespace(config=resolved, provenance=provenance)

    return {"act": loader, "diffusion": loader}


@pytest.mark.parametrize(
    "cell_index, section, field, value",
    [
        (0, "train", "epochs", 99),
        (2, "train", "use_ema", False),
        (0, "run", "run_id", "wrong-run-id"),
    ],
)
def test_return_checkpoint_rejects_wrong_full_cell_config_even_with_refreshed_hash(
    tmp_path, config, cell_index, section, field, value
) -> None:
    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)
    cell = config.cells[cell_index]
    tampered = _resolved_cell_config(cell)
    tampered[section][field] = value
    resolved_path = (
        tmp_path
        / "attempts/12345"
        / str(cell.index)
        / cell.run_id
        / "resolved_config.json"
    )
    resolved_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SweepReturnError, match="resolved|config"):
        verify_sweep_checkpoints(
            tmp_path,
            attempt_id="12345",
            config=config,
            transfer_manifest=transfer,
            loaders=_checkpoint_loaders(config, transfer, configs={cell.run_id: tampered}),
        )


def test_return_checkpoint_rejects_durable_config_tamper_without_checkpoint_update(
    tmp_path, config
) -> None:
    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)
    cell = config.cells[0]
    path = tmp_path / "attempts/12345/0" / cell.run_id / "resolved_config.json"
    payload = json.loads(path.read_text())
    payload["train"]["epochs"] = 99
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SweepReturnError, match="resolved|config"):
        verify_sweep_checkpoints(
            tmp_path,
            attempt_id="12345",
            config=config,
            transfer_manifest=transfer,
            loaders=_checkpoint_loaders(config, transfer),
        )


@pytest.mark.parametrize(
    "field", ["master_dataset_fingerprint_sha256", "action_dataset_fingerprint_sha256"]
)
def test_return_checkpoint_rejects_tampered_source_or_action_fingerprint(
    tmp_path, config, field
) -> None:
    _write_complete_attempt(tmp_path, config)
    transfer = _transfer_manifest(config)
    cell = config.cells[0]
    with pytest.raises(SweepReturnError, match="fingerprint|provenance"):
        verify_sweep_checkpoints(
            tmp_path,
            attempt_id="12345",
            config=config,
            transfer_manifest=transfer,
            loaders=_checkpoint_loaders(
                config,
                transfer,
                provenance_updates={cell.run_id: {field: "0" * 64}},
            ),
        )


def test_sweep_preflight_is_non_isaac_offline_and_dependency_complete(
    tmp_path, config, monkeypatch
) -> None:
    import alexdoor_xas.cluster_sweep.preflight as preflight
    from alexdoor_xas.cluster_pilot.preflight import REQUIRED_IMPORTS

    manifest = {
        "schema": "alexdoor_xas.cluster_sweep_transfer_manifest.v1",
        "source_git": {"commit": "1" * 40},
    }
    dependencies = {
        "python": "3.11.9",
        "numpy": "2.4.6",
        "torch": "2.12.1+cu126",
        "torch_cuda": "12.6",
        "ruff": "ruff 0.15.3",
    }
    dependencies.update({name: "1.0" for name in REQUIRED_IMPORTS})
    dependencies["numpy"] = "2.4.6"
    dependencies["torch"] = "2.12.1+cu126"
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
    assert report["dependencies"]["numpy"] == "2.4.6"
    assert report["dependencies"]["torch"] == "2.12.1+cu126"
    assert report["dependencies"]["torch_cuda"] == "12.6"
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
