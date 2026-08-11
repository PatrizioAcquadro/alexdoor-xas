"""Regression contract for the local Gilbreth N50 compatibility pilot package."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from alexdoor_xas.cluster_pilot import preflight as cluster_preflight
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.preflight import (
    ClusterPreflightError,
    probe_cuda_device,
    run_pure_preflight,
)
from alexdoor_xas.cluster_pilot.returns import (
    ReturnManifestError,
    build_return_manifest,
    return_rsync_template,
    verify_return_checkpoints,
    verify_return_manifest,
)
from alexdoor_xas.cluster_pilot.slurm import render_slurm_script
from alexdoor_xas.cluster_pilot.transfer import (
    PilotTransferError,
    build_transfer_manifest,
    pilot_rsync_file_list,
    pilot_rsync_template,
    secret_problems,
    verify_transfer_manifest,
)
from alexdoor_xas.policies.act import load_act_config
from alexdoor_xas.policies.diffusion import load_diffusion_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cluster_pilot_n50.v1.json"
ATTEMPT_ID = "424242"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture(scope="module")
def config():
    return load_pilot_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def clean_source_state() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "branch": "impl/gilbreth-compatibility-pilot",
        "detached": False,
        "clean_tree": True,
        "commit_time": "2026-07-13T00:00:00+00:00",
    }


@pytest.fixture(scope="module")
def transfer_manifest(config, clean_source_state):
    return build_transfer_manifest(
        REPO_ROOT,
        config,
        source_state=clean_source_state,
    )


def test_pilot_config_freezes_two_short_cells_and_future_contract(config) -> None:
    assert config.schema == "alexdoor_xas.cluster_pilot_config.v1"
    assert config.source_dataset.task == "door_push_alex_v2"
    assert config.source_dataset.version == "v2_pose"
    assert config.source_dataset.obs_preset == "core_door_pose"
    assert config.source_dataset.counts == {"total": 50, "train": 38, "val": 6, "test": 6}
    assert config.training.seed == 0
    assert config.training.device == "cuda"
    assert config.training.epochs == 2
    assert config.training.val_every == 1
    assert config.training.overfit_episodes == 2
    assert config.training.wandb_mode == "offline"
    assert config.slurm.gpus_per_node == 1

    assert [(cell.policy, cell.space, cell.run_id) for cell in config.cells] == [
        ("act", "A2_ee_delta", "pilot_act_a2_n50_seed0"),
        ("diffusion", "A3_obj_rel_ee_delta", "pilot_diffusion_a3_n50_seed0"),
    ]
    assert config.cells[0].overrides["model.chunk_size"] == 40
    assert config.cells[1].overrides["model.horizon"] == 16
    assert config.cells[1].overrides["train.use_ema"] is True
    assert config.cells[1].overrides["train.val_inference_steps"] <= 5

    future = config.future_sweep
    assert future["status"] == "contract_only_do_not_generate_or_launch"
    assert future["training_episode_counts"] == [50, 100, 250, 500]
    assert future["n_definition"] == "number of training episodes"
    assert future["nested_training_subsets"] is True
    assert future["fixed_shared_validation_test"] is True
    assert future["equal_pose_balance"] is True
    assert future["paired_a2_a3_source_episodes"] is True
    assert future["full_matrix_cells"] == 16


def test_pilot_config_rejects_hardware_or_scientific_drift(tmp_path) -> None:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["training"]["epochs"] = 100
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(PilotConfigError, match="epochs"):
        load_pilot_config(path)


def test_environment_spec_freezes_python311_numpy_compatibility_boundary() -> None:
    spec = (REPO_ROOT / "environment" / "gilbreth_pilot_py311.yml").read_text()
    dependency_lines = [line.strip().removeprefix("- ") for line in spec.splitlines()]
    pip_dependencies = [line for line in dependency_lines if "==" in line]

    assert "python=3.11" in dependency_lines
    assert pip_dependencies == [
        "numpy==2.4.6",
        "hydra-core==1.3.3",
        "omegaconf==2.3.1",
        "h5py==3.16.0",
        "matplotlib==3.10.8",
        "diffusers==0.39.0",
        "wandb==0.28.0",
        "pytest==9.1.1",
        "ruff==0.15.3",
    ]
    assert "numpy==2.5.0" not in dependency_lines
    assert "torch==" not in spec.lower()
    assert "isaac" not in spec.lower()


def test_bootstrap_requires_cluster_roots_and_explicit_torch_build() -> None:
    bootstrap = (REPO_ROOT / "scripts" / "bootstrap_gilbreth_pilot.sh").read_text()
    lowered = bootstrap.lower()
    assert "--depot-root" in bootstrap
    assert "--scratch-root" in bootstrap
    assert "--repo-root" in bootstrap
    assert "--torch-spec" in bootstrap
    assert "--torch-index-url" in bootstrap
    assert '[[ -d "$DEPOT_ROOT" ]]' in bootstrap
    assert '[[ -d "$SCRATCH_ROOT" ]]' in bootstrap
    assert 'ENV_PREFIX="$DEPOT_ROOT/' in bootstrap
    assert "torch==" in bootstrap
    assert "--no-deps --editable" in bootstrap
    assert "preflight_cluster_pilot.py" in bootstrap
    assert "isaac sim" not in lowered
    assert "isaac lab" not in lowered


def test_transfer_inventory_is_exactly_a2_a3_split_and_tracked_pilot_files(
    config, transfer_manifest
) -> None:
    entries = transfer_manifest["files"]
    paths = [entry["path"] for entry in entries]
    categories = {entry["category"] for entry in entries}
    assert categories == {"dataset_episode", "dataset_metadata", "split", "pilot_source"}
    assert sum(entry["category"] == "dataset_episode" for entry in entries) == 200
    assert sum(entry["category"] == "dataset_metadata" for entry in entries) == 6
    assert sum(entry["category"] == "split" for entry in entries) == 1
    assert sum(entry["category"] == "pilot_source" for entry in entries) == len(
        config.tracked_transfer_files
    )
    assert (
        "src/alexdoor_xas/cluster_pilot/wandb_publication.py"
        in config.tracked_transfer_files
    )
    assert (
        "knowledge/wiki/implementation_phases/extra-03-gilbreth-compatibility-pilot.md"
        in config.tracked_transfer_files
    )
    assert all("A1_joint_delta" not in path for path in paths)
    assert all("A4_obj_centric_chunk" not in path for path in paths)
    assert all("outputs/local_smoke" not in path for path in paths)
    assert transfer_manifest["dataset"]["counts"] == {
        "total": 50,
        "train": 38,
        "val": 6,
        "test": 6,
    }
    assert set(transfer_manifest["dataset"]["spaces"]) == {
        "A2_ee_delta",
        "A3_obj_rel_ee_delta",
    }
    assert len(transfer_manifest["dataset"]["split_fingerprint_sha256"]) == 64
    for space in ("A2_ee_delta", "A3_obj_rel_ee_delta"):
        assert len(
            transfer_manifest["dataset"]["spaces"][space]["dataset_fingerprint_sha256"]
        ) == 64
        assert (
            transfer_manifest["dataset"]["spaces"][space]["obs_preset"]
            == "core_door_pose"
        )


def test_transfer_manifest_rejects_a1_a4_and_unrelated_artifacts(
    config, clean_source_state, transfer_manifest
) -> None:
    for forbidden in (
        "datasets/door_push_alex_v2/A1_joint_delta/v2_pose/episode_bad.hdf5",
        "datasets/door_push_alex_v2/A4_obj_centric_chunk/v2_pose/episode_bad.hdf5",
        "outputs/local_smoke_n50/checkpoints/best.pt",
    ):
        mutated = copy.deepcopy(transfer_manifest)
        mutated["files"].append(
            {
                "category": "dataset_episode",
                "path": forbidden,
                "size_bytes": 0,
                "sha256": "0" * 64,
            }
        )
        failures = verify_transfer_manifest(
            mutated,
            REPO_ROOT,
            config,
            source_state=clean_source_state,
        )
        assert any("unexpected" in failure or "forbidden" in failure for failure in failures)


def test_transfer_manifest_rejects_missing_extra_and_hash_mismatch(
    config, clean_source_state, transfer_manifest
) -> None:
    missing = copy.deepcopy(transfer_manifest)
    missing["files"].pop(0)
    assert any(
        "missing" in failure
        for failure in verify_transfer_manifest(
            missing, REPO_ROOT, config, source_state=clean_source_state
        )
    )

    extra = copy.deepcopy(transfer_manifest)
    extra["files"].append(copy.deepcopy(extra["files"][0]))
    assert any(
        "duplicate" in failure or "unexpected" in failure
        for failure in verify_transfer_manifest(
            extra, REPO_ROOT, config, source_state=clean_source_state
        )
    )

    mismatched = copy.deepcopy(transfer_manifest)
    mismatched["files"][0]["sha256"] = "0" * 64
    assert any(
        "hash mismatch" in failure
        for failure in verify_transfer_manifest(
            mismatched, REPO_ROOT, config, source_state=clean_source_state
        )
    )


def test_transfer_manifest_rejects_dataset_and_split_fingerprint_mismatch(
    config, clean_source_state, transfer_manifest
) -> None:
    dataset_bad = copy.deepcopy(transfer_manifest)
    dataset_bad["dataset"]["spaces"]["A2_ee_delta"][
        "dataset_fingerprint_sha256"
    ] = "0" * 64
    failures = verify_transfer_manifest(
        dataset_bad, REPO_ROOT, config, source_state=clean_source_state
    )
    assert any("dataset fingerprint" in failure for failure in failures)

    split_bad = copy.deepcopy(transfer_manifest)
    split_bad["dataset"]["split_fingerprint_sha256"] = "0" * 64
    failures = verify_transfer_manifest(
        split_bad, REPO_ROOT, config, source_state=clean_source_state
    )
    assert any("split fingerprint" in failure for failure in failures)


def test_transfer_manifest_requires_clean_matching_source(
    config, clean_source_state, transfer_manifest
) -> None:
    dirty = {**clean_source_state, "clean_tree": False}
    with pytest.raises(PilotTransferError, match="clean"):
        build_transfer_manifest(REPO_ROOT, config, source_state=dirty)

    mismatched = {**clean_source_state, "commit": "2" * 40}
    failures = verify_transfer_manifest(
        transfer_manifest,
        REPO_ROOT,
        config,
        source_state=mismatched,
    )
    assert any("source commit mismatch" in failure for failure in failures)


def test_transfer_manifest_and_rsync_list_are_deterministic(
    config, clean_source_state, transfer_manifest
) -> None:
    rebuilt = build_transfer_manifest(REPO_ROOT, config, source_state=clean_source_state)
    assert rebuilt == transfer_manifest
    first = pilot_rsync_file_list(transfer_manifest)
    assert first == pilot_rsync_file_list(rebuilt)
    assert first[-2:] == [
        "outputs/cluster_pilot_n50/pilot_transfer_manifest.json",
        "outputs/cluster_pilot_n50/rsync-files.txt",
    ]
    command = pilot_rsync_template()
    assert "--partial" in command
    assert "--checksum" in command
    assert "--append-verify" not in command
    assert "--files-from=outputs/cluster_pilot_n50/rsync-files.txt" in command
    assert "<user>@<host>:<remote_root>/" in command


def test_secret_exclusion_covers_paths_and_content() -> None:
    assert secret_problems(".netrc", b"")
    assert secret_problems("configs/pilot.json", b"WANDB_API_KEY=super-secret")
    assert secret_problems("configs/pilot.json", b"-----BEGIN PRIVATE KEY-----")
    assert secret_problems("configs/pilot.json", b"https://user:password@example.test")
    assert secret_problems("configs/pilot.json", b'{"wandb_mode":"offline"}') == []


def test_slurm_renderer_freezes_two_cells_and_durable_fail_closed_flow(
    config, transfer_manifest
) -> None:
    rendered = render_slurm_script(
        config,
        source_commit=transfer_manifest["source_git"]["commit"],
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/alexdoor-results"),
        account="example-account",
        partition="example-partition",
        qos=None,
    )
    assert "#SBATCH --array=0-1%2" in rendered
    assert "#SBATCH --gpus-per-node=1" in rendered
    assert "#SBATCH --gres" not in rendered
    assert "#SBATCH --cpus-per-task=8" in rendered
    assert "#SBATCH --account=example-account" in rendered
    assert "#SBATCH --partition=example-partition" in rendered
    assert "isaaclab.sh" not in rendered.lower()
    assert "import isaac" not in rendered.lower()
    assert "scripts/train_act.py" in rendered
    assert "scripts/train_diffusion.py" in rendered
    assert "dataset.obs_preset=core_door_pose" in rendered
    assert "+wandb.mode=offline" in rendered
    assert "+wandb.dir=$CELL_RUNTIME/wandb" in rendered
    assert "train.device=cuda" in rendered
    assert "run.output_root=" in rendered
    assert "preflight_cluster_pilot.py" in rendered
    assert "verify" in rendered
    assert "completion.json" in rendered
    assert "failure.json" in rendered
    assert "DURABLE_RESULTS_ROOT" in rendered
    assert 'ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"' in rendered
    assert 'CELL_ROOT="$SCRATCH_RUNS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"' in rendered
    assert (
        'PUBLISH_FINAL="$DURABLE_RESULTS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"'
        in rendered
    )
    assert "alexdoor_xas.cluster_pilot_cell_status.v2" in rendered
    assert '"slurm_array_job_id":"%s"' in rendered
    assert '"slurm_array_task_id":"%s"' in rendered
    assert ".tmp" in rendered
    assert "mv" in rendered
    assert "publish_ok" in rendered
    assert "find \"$CELL_RUNTIME/wandb\" -type f" in rendered
    assert "set -Eeuo pipefail" in rendered
    assert "#SBATCH --qos" not in rendered
    assert 'export PATH="$CONDA_PREFIX/bin:$PATH"' in rendered
    python_validation = rendered.index('[[ -x "$CONDA_PREFIX/bin/python" ]]')
    path_prepend = rendered.index('export PATH="$CONDA_PREFIX/bin:$PATH"')
    manifest_check = rendered.index('[[ -f "$MANIFEST" ]]')
    assert python_validation < path_prepend < manifest_check
    assert "bin/activate" not in rendered
    assert "conda activate" not in rendered.lower()
    prefix_python = '"$CONDA_PREFIX/bin/python"'
    assert (
        f"{prefix_python} scripts/build_cluster_pilot_manifest.py verify \\\n"
        in rendered
    )
    assert f"{prefix_python} scripts/preflight_cluster_pilot.py" in rendered
    assert "ENTRYPOINT=scripts/train_act.py" in rendered
    assert "ENTRYPOINT=scripts/train_diffusion.py" in rendered
    assert f'{prefix_python} "$ENTRYPOINT"' in rendered
    assert (
        f"{prefix_python} src/alexdoor_xas/cluster_pilot/wandb_publication.py"
        in rendered
    )
    assert 'cp -a "$CELL_RUNTIME/wandb"' not in rendered


def test_rendered_slurm_polluted_path_smoke_preserves_durable_failure(
    tmp_path, config
) -> None:
    depot_root = tmp_path / "depot"
    scratch_root = tmp_path / "scratch"
    durable_root = depot_root / "durable-results"
    repo_root = depot_root / config.storage.source_checkout_relative
    conda_prefix = depot_root / config.storage.conda_prefix_relative
    polluted_bin = tmp_path / "pyenv-shims"
    source_commit = "1" * 40
    attempt_id = "11279999"

    (repo_root / ".git").mkdir(parents=True)
    manifest_path = repo_root / "outputs/cluster_pilot_n50/pilot_transfer_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n")
    scratch_root.mkdir()

    path_log = tmp_path / "observed-path.txt"
    ruff_log = tmp_path / "resolved-ruff.txt"
    polluted_ruff_marker = tmp_path / "polluted-ruff-ran.txt"
    _write_executable(
        conda_prefix / "bin/python",
        """#!/bin/sh
printf '%s\\n' "$PATH" >> "$SMOKE_PATH_LOG"
case "$1" in
  scripts/build_cluster_pilot_manifest.py)
    exit 0
    ;;
  scripts/preflight_cluster_pilot.py)
    command -v ruff >> "$SMOKE_RUFF_LOG"
    ruff --version
    exit 41
    ;;
  src/alexdoor_xas/cluster_pilot/wandb_publication.py)
    while [ $# -gt 0 ]; do
      if [ "$1" = "--destination" ]; then
        mkdir -p "$2"
        printf '{}\n' > "$2/publication_report.json"
        exit 0
      fi
      shift
    done
    exit 99
    ;;
  *)
    exit 99
    ;;
esac
""",
    )
    _write_executable(
        conda_prefix / "bin/ruff",
        "#!/bin/sh\necho 'ruff active-prefix 0.15.3'\n",
    )
    _write_executable(
        polluted_bin / "ruff",
        "#!/bin/sh\necho ran > \"$SMOKE_POLLUTED_RUFF_MARKER\"\nexit 97\n",
    )
    _write_executable(
        polluted_bin / "git",
        f"""#!/bin/sh
case "$1:$2" in
  rev-parse:HEAD)
    echo {source_commit}
    exit 0
    ;;
  status:--porcelain)
    exit 0
    ;;
  *)
    exit 98
    ;;
esac
""",
    )

    rendered = render_slurm_script(
        config,
        source_commit=source_commit,
        depot_root=depot_root,
        scratch_root=scratch_root,
        durable_results_root=durable_root,
        account="example-account",
        partition="example-partition",
        qos=None,
    )
    script = tmp_path / "pilot.slurm"
    _write_executable(script, rendered)
    environment = {
        **os.environ,
        "PATH": f"{polluted_bin}:/usr/bin:/bin",
        "SLURM_ARRAY_JOB_ID": attempt_id,
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": f"{attempt_id}_0",
        "SMOKE_PATH_LOG": str(path_log),
        "SMOKE_RUFF_LOG": str(ruff_log),
        "SMOKE_POLLUTED_RUFF_MARKER": str(polluted_ruff_marker),
    }
    result = subprocess.run(
        ["/bin/bash", str(script)],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 41, result.stderr
    expected_path_prefix = f"{conda_prefix}/bin:"
    assert path_log.read_text().splitlines()
    assert all(line.startswith(expected_path_prefix) for line in path_log.read_text().splitlines())
    assert ruff_log.read_text().strip() == str(conda_prefix / "bin/ruff")
    assert not polluted_ruff_marker.exists()

    failure_path = (
        durable_root
        / "attempts"
        / attempt_id
        / "0"
        / config.cells[0].run_id
        / "status/failure.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "FAILED"
    assert failure["exit_code"] == 41
    assert failure["attempt"] == {
        "slurm_array_job_id": attempt_id,
        "slurm_array_task_id": "0",
        "run_id": config.cells[0].run_id,
    }


def test_slurm_qos_and_a100_are_only_enabled_explicitly(config) -> None:
    rendered = render_slurm_script(
        config,
        source_commit="1" * 40,
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/results"),
        account="acct",
        partition="a100-example",
        qos="gpu-qos",
        require_a100_80gb=True,
    )
    assert "#SBATCH --qos=gpu-qos" in rendered
    assert "--require-a100-80gb" in rendered


def test_slurm_scheduler_and_array_resources_remain_configurable(config) -> None:
    selected = replace(config, slurm=replace(config.slurm, array_max_concurrent=1))
    rendered = render_slurm_script(
        selected,
        source_commit="1" * 40,
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/results"),
        account="alternate-account",
        partition="alternate-partition",
        qos=None,
        memory="64G",
        cpus_per_task=12,
        wall_time="01:15:00",
    )
    assert "#SBATCH --account=alternate-account" in rendered
    assert "#SBATCH --partition=alternate-partition" in rendered
    assert "#SBATCH --array=0-1%1" in rendered
    assert "#SBATCH --cpus-per-task=12" in rendered
    assert "#SBATCH --mem=64G" in rendered
    assert "#SBATCH --time=01:15:00" in rendered


def test_rendered_hydra_contract_resolves_for_both_training_entrypoints(config) -> None:
    common = [
        f"dataset.task={config.source_dataset.task}",
        f"dataset.version={config.source_dataset.version}",
        f"dataset.obs_preset={config.source_dataset.obs_preset}",
        f"train.seed={config.training.seed}",
        f"train.device={config.training.device}",
        f"train.epochs={config.training.epochs}",
        f"train.val_every={config.training.val_every}",
        f"train.overfit_episodes={config.training.overfit_episodes}",
        "run.output_root=/tmp/pilot-runs",
        f"+wandb.mode={config.training.wandb_mode}",
        "+wandb.dir=/tmp/pilot-wandb",
    ]
    act_cell, diffusion_cell = config.cells
    act = load_act_config(
        [
            f"dataset.space={act_cell.space}",
            f"run.run_id={act_cell.run_id}",
            *common,
            *(f"{key}={value}" for key, value in act_cell.overrides.items()),
        ]
    )
    diffusion = load_diffusion_config(
        [
            f"dataset.space={diffusion_cell.space}",
            f"run.run_id={diffusion_cell.run_id}",
            *common,
            *(
                f"{key}={str(value).lower() if isinstance(value, bool) else value}"
                for key, value in diffusion_cell.overrides.items()
            ),
        ]
    )
    assert act.dataset.obs_preset == "core_door_pose"
    assert act.model.chunk_size == 40
    assert act.wandb_overrides == {"mode": "offline", "dir": "/tmp/pilot-wandb"}
    assert diffusion.dataset.obs_preset == "core_door_pose"
    assert diffusion.model.horizon == 16
    assert diffusion.train.use_ema is True
    assert diffusion.train.val_inference_steps == 5
    assert diffusion.wandb_overrides == {
        "mode": "offline",
        "dir": "/tmp/pilot-wandb",
    }


class _NoCuda:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


def test_live_cuda_probe_cannot_false_pass_without_cuda() -> None:
    with pytest.raises(ClusterPreflightError, match="CUDA"):
        probe_cuda_device(_NoCuda, expected_device_count=1, require_a100_80gb=False)


def test_dependency_inventory_uses_ruff_adjacent_to_active_python(
    tmp_path, monkeypatch
) -> None:
    active_bin = tmp_path / "active-env/bin"
    polluted_bin = tmp_path / "pyenv-shims"
    polluted_marker = tmp_path / "polluted-ruff-ran.txt"
    active_python = active_bin / "python"
    active_python.parent.mkdir(parents=True)
    active_python.touch()
    _write_executable(active_bin / "ruff", "#!/bin/sh\necho 'ruff active-env 0.15.3'\n")
    _write_executable(
        polluted_bin / "ruff",
        f"#!/bin/sh\necho ran > {polluted_marker}\nexit 97\n",
    )
    monkeypatch.setattr(sys, "executable", str(active_python))
    monkeypatch.setattr(cluster_preflight, "REQUIRED_IMPORTS", {})
    monkeypatch.setenv("PATH", str(polluted_bin))

    inventory = cluster_preflight.dependency_inventory()

    assert inventory["ruff"] == "ruff active-env 0.15.3"
    assert not polluted_marker.exists()


@pytest.mark.parametrize("mode", ["missing", "non_executable", "error"])
def test_dependency_inventory_fails_closed_for_invalid_adjacent_ruff(
    tmp_path, monkeypatch, mode
) -> None:
    active_bin = tmp_path / "active-env/bin"
    polluted_bin = tmp_path / "pyenv-shims"
    active_python = active_bin / "python"
    active_python.parent.mkdir(parents=True)
    active_python.touch()
    _write_executable(polluted_bin / "ruff", "#!/bin/sh\necho 'ruff inherited-path'\n")
    adjacent_ruff = active_bin / "ruff"
    if mode == "non_executable":
        adjacent_ruff.write_text("#!/bin/sh\necho should-not-run\n")
    elif mode == "error":
        _write_executable(
            adjacent_ruff,
            "#!/bin/sh\necho 'adjacent ruff failed' >&2\nexit 7\n",
        )
    monkeypatch.setattr(sys, "executable", str(active_python))
    monkeypatch.setattr(cluster_preflight, "REQUIRED_IMPORTS", {})
    monkeypatch.setenv("PATH", str(polluted_bin))

    with pytest.raises(ClusterPreflightError, match="ruff"):
        cluster_preflight.dependency_inventory()


def test_pure_preflight_runs_without_cuda_probe(
    tmp_path, config, clean_source_state, transfer_manifest
) -> None:
    report = run_pure_preflight(
        repo_root=REPO_ROOT,
        config=config,
        manifest=transfer_manifest,
        scratch_output=tmp_path,
        source_state=clean_source_state,
        dependency_probe=lambda: {
            "python": "3.11.9",
            "numpy": "test",
            "hydra": "test",
            "omegaconf": "test",
            "h5py": "test",
            "matplotlib": "test",
            "diffusers": "test",
            "wandb": "test",
            "pytest": "test",
            "ruff": "test",
            "torch": "test",
        },
        isaac_probe=lambda: {},
        checkpoint_probe=lambda path: path.write_bytes(b"checkpoint"),
    )
    assert report["status"] == "PASS"
    assert report["cuda_probe"] == "NOT_RUN"
    assert report["dataset"]["counts"] == {"total": 50, "train": 38, "val": 6, "test": 6}
    assert set(report["dataset"]["spaces"]) == {"A2_ee_delta", "A3_obj_rel_ee_delta"}
    assert not any(name.startswith(("isaac", "omni")) for name in sys.modules)


def _attempt_run_root(root: Path, cell, attempt_id: str = ATTEMPT_ID) -> Path:
    return root / "attempts" / attempt_id / str(cell.index) / cell.run_id


def _make_return_tree(root: Path, config, *, attempt_id: str = ATTEMPT_ID) -> None:
    for cell in config.cells:
        run = _attempt_run_root(root, cell, attempt_id)
        required = {
            "checkpoints/best.pt": b"best",
            "checkpoints/last.pt": b"last",
            "logs/train_log.json": b"{}\n",
            "metrics/open_loop.json": b"{}\n",
            "resolved_config.json": json.dumps(
                {"dataset": {"space": cell.space}, "run": {"run_id": cell.run_id}}
            ).encode(),
            "wandb/offline-run-test/run-test.wandb": b"offline",
            "environment/environment_inventory.json": b"{}\n",
            "environment/requirements.lock": b"numpy==test\n",
            "slurm/stdout.log": b"stdout\n",
            "slurm/stderr.log": b"",
            "status/completion.json": json.dumps(
                {
                    "schema": "alexdoor_xas.cluster_pilot_cell_status.v2",
                    "status": "COMPLETED",
                    "run_id": cell.run_id,
                    "policy": cell.policy,
                    "space": cell.space,
                    "exit_code": 0,
                    "source_git_commit": "1" * 40,
                    "attempt": {
                        "slurm_array_job_id": attempt_id,
                        "slurm_array_task_id": str(cell.index),
                        "run_id": cell.run_id,
                    },
                }
            ).encode(),
        }
        for relative, content in required.items():
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def test_return_manifest_covers_and_verifies_required_artifacts(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config)
    manifest = build_return_manifest(
        results, config, transfer_manifest, attempt_id=ATTEMPT_ID
    )
    assert manifest["schema"] == "alexdoor_xas.cluster_pilot_return_manifest.v2"
    assert manifest["source_git_commit"] == transfer_manifest["source_git"]["commit"]
    assert manifest["provenance"]["slurm_array_job_id"] == ATTEMPT_ID
    assert verify_return_manifest(
        manifest, results, config, attempt_id=ATTEMPT_ID
    ) == []
    paths = {entry["path"] for entry in manifest["files"]}
    for cell in config.cells:
        prefix = f"attempts/{ATTEMPT_ID}/{cell.index}/{cell.run_id}"
        for suffix in (
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "logs/train_log.json",
            "metrics/open_loop.json",
            "resolved_config.json",
            "environment/environment_inventory.json",
            "environment/requirements.lock",
            "slurm/stdout.log",
            "slurm/stderr.log",
            "status/completion.json",
        ):
            assert f"{prefix}/{suffix}" in paths
        assert any(path.startswith(f"{prefix}/wandb/") for path in paths)

    command = return_rsync_template(ATTEMPT_ID)
    assert "--partial" in command
    assert "--checksum" in command
    assert "--append-verify" not in command
    assert (
        "--files-from=:<remote_results_root>/.pilot_return/attempts/"
        f"{ATTEMPT_ID}/return-files.txt" in command
    )
    assert "<user>@<host>:<remote_results_root>/" in command


def test_return_manifest_rejects_tampering_and_checkpoint_loader_covers_both_policies(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config)
    manifest = build_return_manifest(
        results, config, transfer_manifest, attempt_id=ATTEMPT_ID
    )
    (_attempt_run_root(results, config.cells[0]) / "checkpoints" / "best.pt").write_bytes(
        b"tampered"
    )
    assert any(
        "hash mismatch" in problem
        for problem in verify_return_manifest(
            manifest, results, config, attempt_id=ATTEMPT_ID
        )
    )

    _make_return_tree(results, config)
    status_path = _attempt_run_root(results, config.cells[0]) / "status" / "completion.json"
    status = json.loads(status_path.read_text())
    status["source_git_commit"] = "2" * 40
    status_path.write_text(json.dumps(status))
    with pytest.raises(ReturnManifestError, match="source commit mismatch"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )

    _make_return_tree(results, config)
    calls: list[tuple[str, Path]] = []
    loaded = verify_return_checkpoints(
        results,
        config,
        attempt_id=ATTEMPT_ID,
        loaders={
            "act": lambda path: calls.append(("act", path)) or {"ok": True},
            "diffusion": lambda path: calls.append(("diffusion", path)) or {"ok": True},
        },
    )
    assert loaded == {
        "pilot_act_a2_n50_seed0": "PASS",
        "pilot_diffusion_a3_n50_seed0": "PASS",
    }
    assert [policy for policy, _ in calls] == ["act", "diffusion"]


def test_return_manifest_requires_one_explicit_nonstale_attempt(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config, attempt_id="111111")

    with pytest.raises(ReturnManifestError, match="selected durable attempt"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )

    for cell in config.cells:
        legacy = results / cell.run_id
        legacy.mkdir(parents=True)
        (legacy / "stale.txt").write_text("stale\n")
    with pytest.raises(ReturnManifestError, match="selected durable attempt"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )


def test_return_manifest_rejects_mixed_attempt_task_and_run_identity(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config)
    first = config.cells[0]
    status_path = _attempt_run_root(results, first) / "status" / "completion.json"
    status = json.loads(status_path.read_text())

    status["attempt"]["slurm_array_job_id"] = "111111"
    status_path.write_text(json.dumps(status))
    with pytest.raises(ReturnManifestError, match="attempt identity mismatch"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )

    _make_return_tree(results, config)
    status = json.loads(status_path.read_text())
    status["attempt"]["slurm_array_task_id"] = "1"
    status_path.write_text(json.dumps(status))
    with pytest.raises(ReturnManifestError, match="attempt identity mismatch"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )

    _make_return_tree(results, config)
    status = json.loads(status_path.read_text())
    status["attempt"]["run_id"] = config.cells[1].run_id
    status_path.write_text(json.dumps(status))
    with pytest.raises(ReturnManifestError, match="attempt identity mismatch"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )


def test_return_manifest_verification_rejects_wrong_selected_attempt(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config)
    manifest = build_return_manifest(
        results, config, transfer_manifest, attempt_id=ATTEMPT_ID
    )
    failures = verify_return_manifest(
        manifest, results, config, attempt_id="111111"
    )
    assert any("selected attempt mismatch" in failure for failure in failures)


def test_return_manifest_keeps_fail_closed_symlink_rejection(
    tmp_path, config, transfer_manifest
) -> None:
    results = tmp_path / "results"
    _make_return_tree(results, config)
    run = _attempt_run_root(results, config.cells[0])
    target = run / "wandb/offline-run-test/debug.log"
    target.write_text("debug\n")
    (run / "wandb/debug.log").symlink_to(target)

    with pytest.raises(ReturnManifestError, match="symlinks are forbidden"):
        build_return_manifest(
            results, config, transfer_manifest, attempt_id=ATTEMPT_ID
        )
