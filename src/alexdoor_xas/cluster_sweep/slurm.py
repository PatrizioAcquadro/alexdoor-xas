"""Fail-closed 16-cell Slurm renderer for the non-simulator training sweep."""

# ruff: noqa: E501 - generated shell lines remain whole for auditability.

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .config import SweepCell, SweepConfig

SAFE_DIRECTIVE_RE = re.compile(r"^[A-Za-z0-9._/+:-]+$")


class SweepSlurmError(ValueError):
    """Raised when render-time paths or scheduler values are unsafe."""


def render_sweep_slurm_script(
    config: SweepConfig,
    *,
    source_commit: str,
    depot_root: Path,
    scratch_root: Path,
    durable_results_root: Path,
    account: str,
    partition: str,
    qos: str | None,
    memory: str | None = None,
    cpus_per_task: int | None = None,
    wall_time: str | None = None,
    require_a100_80gb: bool | None = None,
) -> str:
    """Render the stable 0-15 array without allocating or submitting resources."""
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise SweepSlurmError("source_commit must be a full 40-character SHA-1")
    depot = _absolute("depot_root", depot_root)
    scratch = _absolute("scratch_root", scratch_root)
    durable = _absolute("durable_results_root", durable_results_root)
    account_value = _directive("account", account)
    partition_value = _directive("partition", partition)
    qos_value = _directive("qos", qos) if qos is not None else None
    memory_value = _directive("memory", memory or config.slurm.memory)
    wall_time_value = _directive("wall_time", wall_time or config.slurm.wall_time)
    cpus = cpus_per_task or config.slurm.cpus_per_task
    if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus <= 0:
        raise SweepSlurmError("cpus_per_task must be a positive integer")
    concurrency = config.slurm.array_max_concurrent
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise SweepSlurmError("array concurrency must be a positive integer")
    if config.slurm.gpus_per_task != 1:
        raise SweepSlurmError("the sweep requires exactly one GPU per task")
    require_a100 = (
        config.slurm.require_a100_80gb
        if require_a100_80gb is None
        else require_a100_80gb
    )
    if not isinstance(require_a100, bool):
        raise SweepSlurmError("require_a100_80gb must be boolean")

    repo_root = depot / config.storage.source_checkout_relative
    conda_prefix = depot / config.storage.conda_prefix_relative
    scratch_runs = scratch / config.storage.scratch_runs_relative
    slurm_logs = scratch_runs / "slurm"
    lines = [
        "#!/usr/bin/env bash",
        "# Generated from configs/cluster_sweep.v1.json; do not hand-edit.",
        "#SBATCH --job-name=alexdoor-scale-sweep",
        f"#SBATCH --account={account_value}",
        f"#SBATCH --partition={partition_value}",
        f"#SBATCH --array=0-15%{concurrency}",
        "#SBATCH --gpus-per-node=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory_value}",
        f"#SBATCH --time={wall_time_value}",
    ]
    if qos_value is not None:
        lines.append(f"#SBATCH --qos={qos_value}")
    lines.extend(
        [
            f"#SBATCH --output={slurm_logs}/%A_%a.out",
            f"#SBATCH --error={slurm_logs}/%A_%a.err",
            "",
            "set -Eeuo pipefail",
            "umask 077",
            "",
            f"SOURCE_COMMIT={shlex.quote(source_commit)}",
            f"DEPOT_ROOT={shlex.quote(str(depot))}",
            f"SCRATCH_ROOT={shlex.quote(str(scratch))}",
            f"REPO_ROOT={shlex.quote(str(repo_root))}",
            f"CONDA_PREFIX={shlex.quote(str(conda_prefix))}",
            f"SCRATCH_RUNS_ROOT={shlex.quote(str(scratch_runs))}",
            f"DURABLE_RESULTS_ROOT={shlex.quote(str(durable))}",
            f"PARTITION={shlex.quote(partition_value)}",
            'ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"',
            'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"',
            'MANIFEST="$REPO_ROOT/outputs/cluster_sweep/sweep_transfer_manifest.json"',
            "",
            '[[ "$ATTEMPT_ID" =~ ^[0-9]+$ ]] || { echo "invalid array job ID" >&2; exit 19; }',
            '[[ "$TASK_ID" =~ ^[0-9]+$ ]] || { echo "invalid array task ID" >&2; exit 19; }',
            '[[ -d "$DEPOT_ROOT" && -d "$SCRATCH_ROOT" ]] || { echo "missing storage root" >&2; exit 20; }',
            '[[ -d "$REPO_ROOT/.git" ]] || { echo "missing source checkout" >&2; exit 21; }',
            '[[ -x "$CONDA_PREFIX/bin/python" ]] || { echo "missing environment Python" >&2; exit 22; }',
            'export PATH="$CONDA_PREFIX/bin:$PATH"',
            '[[ -f "$MANIFEST" ]] || { echo "missing sweep transfer manifest" >&2; exit 23; }',
            'mkdir -p "$SCRATCH_RUNS_ROOT" "$DURABLE_RESULTS_ROOT"',
            'cd "$REPO_ROOT"',
            '[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 24; }',
            '[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 25; }',
            "",
            'case "$TASK_ID" in',
            *_render_cells(config.cells),
            '  *) echo "unexpected array task: $TASK_ID" >&2; exit 26 ;;',
            "esac",
            "",
            'CELL_ROOT="$SCRATCH_RUNS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"',
            'CELL_RUNTIME="$CELL_ROOT/runtime"',
            'RUN_OUTPUT_ROOT="$CELL_ROOT/output"',
            'RUN_DIR="$RUN_OUTPUT_ROOT/$EXPERIMENT/$RUN_ID"',
            'PUBLISH_FINAL="$DURABLE_RESULTS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"',
            '[[ ! -e "$CELL_ROOT" ]] || { echo "scratch attempt already exists" >&2; exit 27; }',
            '[[ ! -e "$PUBLISH_FINAL" ]] || { echo "durable attempt already exists" >&2; exit 28; }',
            'mkdir -p "$CELL_RUNTIME/environment" "$CELL_RUNTIME/slurm" "$CELL_RUNTIME/status" "$CELL_RUNTIME/wandb"',
            "",
            "write_status() {",
            "  local code=$1 status_name status_value",
            "  if [[ $code -eq 0 ]]; then status_name=completion.json; status_value=COMPLETED; else status_name=failure.json; status_value=FAILED; fi",
            '  rm -f "$CELL_RUNTIME/status/completion.json" "$CELL_RUNTIME/status/failure.json"',
            '  local temporary="$CELL_RUNTIME/status/.${status_name}.tmp"',
            "  printf '{\"schema\":\"alexdoor_xas.cluster_sweep_cell_status.v1\",\"status\":\"%s\",\"run_id\":\"%s\",\"policy\":\"%s\",\"space\":\"%s\",\"view_id\":\"%s\",\"exit_code\":%d,\"source_git_commit\":\"%s\",\"attempt\":{\"slurm_array_job_id\":\"%s\",\"slurm_array_task_id\":\"%s\",\"run_id\":\"%s\"}}\\n' \\",
            '    "$status_value" "$RUN_ID" "$POLICY" "$SPACE" "$VIEW_ID" "$code" "$SOURCE_COMMIT" "$ATTEMPT_ID" "$TASK_ID" "$RUN_ID" > "$temporary"',
            '  mv "$temporary" "$CELL_RUNTIME/status/$status_name"',
            "}",
            "",
            "publish_result() {",
            "  local original_code=$?",
            "  trap - EXIT",
            "  set +e",
            "  local final_code=$original_code publish_ok=1",
            '  write_status "$original_code" || { final_code=90; publish_ok=0; }',
            '  local publish_parent="$DURABLE_RESULTS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID"',
            '  local publish_tmp="$publish_parent/.${RUN_ID}.${SLURM_JOB_ID:-$ATTEMPT_ID.$TASK_ID}.tmp"',
            '  local publish_claim="$publish_parent/.${RUN_ID}.claim"',
            '  mkdir -p "$publish_parent" || publish_ok=0',
            '  if [[ -e "$publish_tmp" || -e "$PUBLISH_FINAL" || -e "$publish_claim" ]]; then publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then mkdir "$publish_claim" || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then mkdir "$publish_tmp" || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 && -d "$RUN_DIR" ]]; then cp -a "$RUN_DIR"/. "$publish_tmp"/ || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then cp -a "$CELL_RUNTIME/environment" "$CELL_RUNTIME/slurm" "$CELL_RUNTIME/status" "$publish_tmp"/ || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then "$CONDA_PREFIX/bin/python" src/alexdoor_xas/cluster_pilot/wandb_publication.py --source "$CELL_RUNTIME/wandb" --destination "$publish_tmp/wandb" || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then [[ -s "$publish_tmp/wandb/publication_report.json" ]] || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 1 ]]; then mv -T "$publish_tmp" "$PUBLISH_FINAL" || publish_ok=0; fi',
            '  if [[ $publish_ok -eq 0 ]]; then final_code=92; fi',
            '  rmdir "$publish_claim" 2>/dev/null || true',
            '  if [[ $final_code -ne $original_code ]]; then write_status "$final_code" || true; fi',
            '  exit "$final_code"',
            "}",
            "trap publish_result EXIT",
            "",
            "COMMON_OVERRIDES=(",
            f'  "dataset.task={config.dataset.task}"',
            f'  "dataset.version={config.dataset.master_version}"',
            '  "dataset.view_id=$VIEW_ID"',
            f'  "dataset.obs_preset={config.dataset.obs_preset}"',
            f'  "train.seed={config.training.seed}"',
            f'  "train.device={config.training.device}"',
            '  "train.overfit_episodes=null"',
            '  "run.run_id=$RUN_ID"',
            '  "run.output_root=$RUN_OUTPUT_ROOT"',
            f'  "+wandb.mode={config.training.wandb_mode}"',
            '  "+wandb.dir=$CELL_RUNTIME/wandb"',
            ")",
            "",
            '"$CONDA_PREFIX/bin/python" scripts/build_cluster_sweep_manifest.py verify --config configs/cluster_sweep.v1.json --manifest "$MANIFEST"',
            "PREFLIGHT_ARGS=(--config configs/cluster_sweep.v1.json --manifest \"$MANIFEST\" --scratch-output \"$CELL_RUNTIME\" --report \"$CELL_RUNTIME/environment/preflight_report.json\" --environment-dir \"$CELL_RUNTIME/environment\" --live-cuda --expected-device-count 1 --requested-partition \"$PARTITION\")",
        ]
    )
    if require_a100:
        lines.append("PREFLIGHT_ARGS+=(--require-a100-80gb)")
    lines.extend(
        [
            "",
            "set +e",
            '"$CONDA_PREFIX/bin/python" scripts/preflight_cluster_sweep.py "${PREFLIGHT_ARGS[@]}" > "$CELL_RUNTIME/slurm/stdout.log" 2> "$CELL_RUNTIME/slurm/stderr.log"',
            "run_code=$?",
            "if [[ $run_code -eq 0 ]]; then",
            '  "$CONDA_PREFIX/bin/python" "$ENTRYPOINT" "dataset.space=$SPACE" "${COMMON_OVERRIDES[@]}" "${POLICY_OVERRIDES[@]}" >> "$CELL_RUNTIME/slurm/stdout.log" 2>> "$CELL_RUNTIME/slurm/stderr.log"',
            "  run_code=$?",
            "fi",
            "set -e",
            'cat "$CELL_RUNTIME/slurm/stdout.log"',
            'cat "$CELL_RUNTIME/slurm/stderr.log" >&2',
            '[[ $run_code -eq 0 ]] || exit "$run_code"',
            'for required in checkpoints/best.pt checkpoints/last.pt logs/train_log.json metrics/open_loop.json resolved_config.json; do [[ -s "$RUN_DIR/$required" ]] || { echo "missing completed artifact: $required" >&2; exit 93; }; done',
            '[[ -s "$CELL_RUNTIME/environment/environment_inventory.json" ]] || exit 94',
            '[[ -s "$CELL_RUNTIME/environment/requirements.lock" ]] || exit 95',
            '[[ -n "$(find "$CELL_RUNTIME/wandb" -type f -print -quit)" ]] || exit 96',
            "exit 0",
            "",
        ]
    )
    return "\n".join(lines)


def _render_cells(cells: tuple[SweepCell, ...]) -> list[str]:
    experiments = {"act": "act_door_push", "diffusion": "diffusion_door_push"}
    lines: list[str] = []
    for cell in cells:
        lines.extend(
            [
                f"  {cell.index})",
                f"    POLICY={shlex.quote(cell.policy)}",
                f"    SPACE={shlex.quote(cell.space)}",
                f"    VIEW_ID={shlex.quote(cell.view_id)}",
                f"    RUN_ID={shlex.quote(cell.run_id)}",
                f"    ENTRYPOINT={shlex.quote(cell.entrypoint)}",
                f"    EXPERIMENT={shlex.quote(experiments[cell.policy])}",
                "    POLICY_OVERRIDES=(",
            ]
        )
        for key, value in cell.overrides.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            lines.append(f"      {shlex.quote(f'{key}={rendered}')}")
        lines.extend(["    )", "    ;;"])
    return lines


def _absolute(name: str, value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(c in str(path) for c in ("\n", "\r", "\0")):
        raise SweepSlurmError(f"{name} must be a safe absolute path")
    return path


def _directive(name: str, value: str) -> str:
    if not isinstance(value, str) or SAFE_DIRECTIVE_RE.fullmatch(value) is None:
        raise SweepSlurmError(f"{name} is required and contains unsupported characters")
    return value


__all__ = ["SweepSlurmError", "render_sweep_slurm_script"]
