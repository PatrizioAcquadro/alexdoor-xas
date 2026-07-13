"""Regression contract for symlink-free durable W&B publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from alexdoor_xas.cluster_pilot.config import load_pilot_config
from alexdoor_xas.cluster_pilot.slurm import render_slurm_script
from alexdoor_xas.cluster_pilot.wandb_publication import (
    PUBLICATION_REPORT_NAME,
    PUBLICATION_SCHEMA,
    WandbPublicationError,
    publish_wandb_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cluster_pilot_n50.v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_tree(root: Path) -> dict[str, tuple[str, int, int, str]]:
    snapshot: dict[str, tuple[str, int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            snapshot[relative] = (
                "symlink",
                stat.S_IMODE(metadata.st_mode),
                metadata.st_mtime_ns,
                os.readlink(path),
            )
        elif path.is_file():
            snapshot[relative] = (
                "file",
                stat.S_IMODE(metadata.st_mode),
                metadata.st_mtime_ns,
                _sha256(path),
            )
        elif path.is_dir():
            snapshot[relative] = (
                "directory",
                stat.S_IMODE(metadata.st_mode),
                metadata.st_mtime_ns,
                "",
            )
        else:
            snapshot[relative] = (
                "special",
                stat.S_IMODE(metadata.st_mode),
                metadata.st_mtime_ns,
                "",
            )
    return snapshot


def _make_observed_wandb_tree(root: Path, *, unsafe: str | None = None) -> dict[str, Path]:
    run = root / "wandb/offline-run-20260713_143612-test1234"
    logs = run / "logs"
    cache_logs = root / "cache/wandb/logs"
    logs.mkdir(parents=True)
    cache_logs.mkdir(parents=True)

    targets = {
        "debug": logs / "debug.log",
        "internal": logs / "debug-internal.log",
        "core": cache_logs / "core-debug-20260713_143613.log",
    }
    targets["debug"].write_text("sdk debug\n")
    targets["internal"].write_text("sdk internal debug\n")
    targets["core"].write_text("core debug\n")
    (run / "run-test1234.wandb").write_bytes(b"offline-run")

    os.chmod(targets["debug"], 0o640)
    os.utime(targets["debug"], ns=(1_700_000_000_123_456_789,) * 2)
    (root / "wandb/debug.log").symlink_to(
        "offline-run-20260713_143612-test1234/logs/debug.log"
    )
    (root / "wandb/debug-internal.log").symlink_to(
        "offline-run-20260713_143612-test1234/logs/debug-internal.log"
    )
    (logs / "debug-core.log").symlink_to(targets["core"])
    (root / "wandb/latest-run").symlink_to(run.name, target_is_directory=True)

    if unsafe == "escape":
        outside = root.parent / "outside.log"
        outside.write_text("outside\n")
        (root / "wandb/escape.log").symlink_to(outside)
    elif unsafe == "broken":
        (root / "wandb/broken.log").symlink_to(root / "missing.log")
    elif unsafe == "directory":
        (root / "wandb/unexpected-run").symlink_to(run, target_is_directory=True)
    elif unsafe == "fifo":
        os.mkfifo(root / "wandb/events.fifo")
    return targets


def test_observed_wandb_links_are_materialized_without_mutating_source(tmp_path) -> None:
    source = tmp_path / "runtime/wandb"
    targets = _make_observed_wandb_tree(source)
    before = _snapshot_tree(source)
    destination = tmp_path / "publication/wandb"
    destination.parent.mkdir(parents=True)

    report = publish_wandb_tree(source, destination)

    materialized = {
        "wandb/debug.log": targets["debug"],
        "wandb/debug-internal.log": targets["internal"],
        "wandb/offline-run-20260713_143612-test1234/logs/debug-core.log": targets[
            "core"
        ],
    }
    for relative, target in materialized.items():
        published = destination / relative
        assert published.is_file()
        assert not published.is_symlink()
        assert _sha256(published) == _sha256(target)
    assert stat.S_IMODE((destination / "wandb/debug.log").stat().st_mode) == 0o640
    assert (destination / "wandb/debug.log").stat().st_mtime_ns == targets[
        "debug"
    ].stat().st_mtime_ns
    assert not (destination / "wandb/latest-run").exists()
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    assert _snapshot_tree(source) == before

    report_path = destination / PUBLICATION_REPORT_NAME
    assert json.loads(report_path.read_text()) == report
    assert report["schema"] == PUBLICATION_SCHEMA
    assert [entry["path"] for entry in report["materialized_symlinks"]] == sorted(
        materialized
    )
    assert all(len(entry["sha256"]) == 64 for entry in report["materialized_symlinks"])
    assert report["omitted_latest_run_symlinks"] == ["wandb/latest-run"]
    assert report["destination_contains_symlinks"] is False
    assert report["destination_symlink_count"] == 0
    assert all("/tmp/" not in json.dumps(entry) for entry in report["materialized_symlinks"])


@pytest.mark.parametrize(
    ("unsafe", "message"),
    [
        ("escape", "escapes source W&B tree"),
        ("broken", "broken symlink"),
        ("directory", "unexpected directory symlink"),
        ("fifo", "special file"),
    ],
)
def test_unsafe_wandb_trees_are_rejected_before_destination_creation(
    tmp_path, unsafe: str, message: str
) -> None:
    source = tmp_path / "runtime/wandb"
    _make_observed_wandb_tree(source, unsafe=unsafe)
    before = _snapshot_tree(source)
    destination = tmp_path / "publication/wandb"
    destination.parent.mkdir(parents=True)

    with pytest.raises(WandbPublicationError, match=message):
        publish_wandb_tree(source, destination)

    assert not destination.exists()
    assert _snapshot_tree(source) == before


def test_latest_run_is_validated_before_omission(tmp_path) -> None:
    source = tmp_path / "runtime/wandb"
    _make_observed_wandb_tree(source)
    latest = source / "wandb/latest-run"
    latest.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    latest.symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "publication/wandb"
    destination.parent.mkdir(parents=True)

    with pytest.raises(WandbPublicationError, match="escapes source W&B tree"):
        publish_wandb_tree(source, destination)

    assert not destination.exists()


def test_existing_destination_is_never_overwritten(tmp_path) -> None:
    source = tmp_path / "runtime/wandb"
    _make_observed_wandb_tree(source)
    destination = tmp_path / "publication/wandb"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep\n")

    with pytest.raises(WandbPublicationError, match="destination already exists"):
        publish_wandb_tree(source, destination)

    assert marker.read_text() == "keep\n"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def _make_generated_script_fixture(tmp_path: Path, *, unsafe: str | None) -> tuple[Path, ...]:
    config = load_pilot_config(CONFIG_PATH)
    depot = tmp_path / "depot"
    scratch = tmp_path / "scratch"
    durable = depot / "durable-results"
    repo = depot / config.storage.source_checkout_relative
    conda = depot / config.storage.conda_prefix_relative
    source_commit = "1" * 40
    (repo / ".git").mkdir(parents=True)
    (repo / "outputs/cluster_pilot_n50").mkdir(parents=True)
    (repo / "outputs/cluster_pilot_n50/pilot_transfer_manifest.json").write_text("{}\n")
    scratch.mkdir()
    (conda / "bin").mkdir(parents=True)
    (conda / "bin/python").symlink_to(sys.executable)
    _write_executable(
        conda / "bin/git",
        f"""#!/bin/sh
case "$1:$2" in
  rev-parse:HEAD) echo {source_commit} ;;
  status:--porcelain) exit 0 ;;
  *) exit 98 ;;
esac
""",
    )
    _write_executable(
        repo / "scripts/build_cluster_pilot_manifest.py",
        "#!/usr/bin/env python\n",
    )
    _write_executable(
        repo / "scripts/preflight_cluster_pilot.py",
        """#!/usr/bin/env python
import pathlib
import sys

args = sys.argv[1:]
environment = pathlib.Path(args[args.index("--environment-dir") + 1])
environment.mkdir(parents=True, exist_ok=True)
(environment / "environment_inventory.json").write_text("{}\\n")
(environment / "requirements.lock").write_text("numpy==test\\n")
""",
    )
    unsafe_literal = repr(unsafe)
    _write_executable(
        repo / "scripts/train_act.py",
        f"""#!/usr/bin/env python
import os
import pathlib
import sys

values = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
output = pathlib.Path(values["run.output_root"]) / "act_door_push" / values["run.run_id"]
for relative in (
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/train_log.json",
    "metrics/open_loop.json",
    "resolved_config.json",
):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("payload\\n")
wandb = pathlib.Path(values["+wandb.dir"])
run = wandb / "wandb/offline-run-20260713_143612-test1234"
logs = run / "logs"
cache = wandb / "cache/wandb/logs"
logs.mkdir(parents=True)
cache.mkdir(parents=True)
(logs / "debug.log").write_text("debug\\n")
(logs / "debug-internal.log").write_text("internal\\n")
core = cache / "core-debug-20260713_143613.log"
core.write_text("core\\n")
(run / "run-test1234.wandb").write_bytes(b"offline")
(wandb / "wandb/debug.log").symlink_to(
    "offline-run-20260713_143612-test1234/logs/debug.log"
)
(wandb / "wandb/debug-internal.log").symlink_to(
    "offline-run-20260713_143612-test1234/logs/debug-internal.log"
)
(logs / "debug-core.log").symlink_to(core)
(wandb / "wandb/latest-run").symlink_to(run.name, target_is_directory=True)
if {unsafe_literal} == "escape":
    outside = wandb.parent / "outside.log"
    outside.write_text("outside\\n")
    (wandb / "wandb/escape.log").symlink_to(outside)
""",
    )
    sanitizer = repo / "src/alexdoor_xas/cluster_pilot/wandb_publication.py"
    sanitizer.parent.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "src/alexdoor_xas/cluster_pilot/wandb_publication.py",
        sanitizer,
    )
    rendered = render_slurm_script(
        config,
        source_commit=source_commit,
        depot_root=depot,
        scratch_root=scratch,
        durable_results_root=durable,
        account="example-account",
        partition="example-partition",
        qos=None,
    )
    script = tmp_path / "pilot.slurm"
    _write_executable(script, rendered)
    return script, repo, scratch, durable, conda


def _run_generated_script(
    script: Path, repo: Path, conda: Path, attempt_id: str
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PATH": f"{conda / 'bin'}:/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SLURM_ARRAY_JOB_ID": attempt_id,
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": f"{attempt_id}_0",
    }
    return subprocess.run(
        ["/bin/bash", str(script)],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generated_script_atomically_publishes_realistic_symlink_free_wandb(
    tmp_path,
) -> None:
    script, repo, _scratch, durable, conda = _make_generated_script_fixture(
        tmp_path, unsafe=None
    )
    result = _run_generated_script(script, repo, conda, "11280001")
    assert result.returncode == 0, result.stderr

    config = load_pilot_config(CONFIG_PATH)
    final = durable / "attempts/11280001/0" / config.cells[0].run_id
    assert (final / "status/completion.json").is_file()
    assert (final / "wandb/wandb/debug.log").is_file()
    assert not (final / "wandb/wandb/debug.log").is_symlink()
    debug_core = (
        final
        / "wandb/wandb/offline-run-20260713_143612-test1234/logs/debug-core.log"
    )
    assert debug_core.is_file()
    assert not (final / "wandb/wandb/latest-run").exists()
    assert (final / f"wandb/{PUBLICATION_REPORT_NAME}").is_file()
    assert not any(path.is_symlink() for path in final.rglob("*"))

    duplicate = _run_generated_script(script, repo, conda, "11280001")
    assert duplicate.returncode != 0
    retry = _run_generated_script(script, repo, conda, "11280002")
    assert retry.returncode == 0, retry.stderr
    retry_final = durable / "attempts/11280002/0" / config.cells[0].run_id
    assert retry_final.is_dir()
    assert final.is_dir()


def test_unsafe_wandb_tree_prevents_final_atomic_publication(tmp_path) -> None:
    script, repo, scratch, durable, conda = _make_generated_script_fixture(
        tmp_path, unsafe="escape"
    )
    result = _run_generated_script(script, repo, conda, "11280003")
    assert result.returncode == 92

    config = load_pilot_config(CONFIG_PATH)
    final = durable / "attempts/11280003/0" / config.cells[0].run_id
    assert not final.exists()
    failure = (
        scratch
        / "alexdoor-xas/cluster_pilot_n50/attempts/11280003/0"
        / config.cells[0].run_id
        / "runtime/status/failure.json"
    )
    assert json.loads(failure.read_text())["exit_code"] == 92
    assert "escapes source W&B tree" in result.stderr


def test_generated_script_uses_activation_free_conda_python_sanitizer() -> None:
    config = load_pilot_config(CONFIG_PATH)
    rendered = render_slurm_script(
        config,
        source_commit="1" * 40,
        depot_root=Path("/depot/example"),
        scratch_root=Path("/scratch/example"),
        durable_results_root=Path("/depot/example/results"),
        account="account",
        partition="partition",
        qos=None,
    )
    invocation = (
        '"$CONDA_PREFIX/bin/python" '
        "src/alexdoor_xas/cluster_pilot/wandb_publication.py"
    )
    assert invocation in rendered
    assert "conda activate" not in rendered.lower()
    assert "bin/activate" not in rendered
