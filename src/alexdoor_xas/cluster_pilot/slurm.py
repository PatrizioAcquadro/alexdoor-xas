"""Deterministic two-cell Slurm renderer for the Gilbreth compatibility pilot."""

# ruff: noqa: E501 - generated shell command lines remain intact for auditability.

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .config import PilotCell, PilotConfig

SAFE_DIRECTIVE_RE = re.compile(r"^[A-Za-z0-9._/+:-]+$")


class SlurmRenderError(ValueError):
    """Raised when required scheduler or storage values are missing or unsafe."""


def render_slurm_script(
    config: PilotConfig,
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
    """Render a fail-closed array script without submitting it."""
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise SlurmRenderError("source_commit must be a full 40-character SHA-1")
    depot = _absolute_path("depot_root", depot_root)
    scratch = _absolute_path("scratch_root", scratch_root)
    durable = _absolute_path("durable_results_root", durable_results_root)
    account = _directive("account", account)
    partition = _directive("partition", partition)
    qos = _directive("qos", qos) if qos is not None else None
    selected_memory = _directive("memory", memory or config.slurm.memory)
    selected_wall_time = _directive("wall_time", wall_time or config.slurm.wall_time)
    selected_cpus = cpus_per_task or config.slurm.cpus_per_task
    if not isinstance(selected_cpus, int) or selected_cpus <= 0:
        raise SlurmRenderError("cpus_per_task must be a positive integer")
    if config.slurm.gpus_per_node != 1:
        raise SlurmRenderError("the pilot must request exactly one GPU per node")
    concurrency = config.slurm.array_max_concurrent
    if concurrency < 1 or concurrency > 2:
        raise SlurmRenderError("array concurrency must be between one and two")
    require_a100 = (
        config.slurm.require_a100_80gb
        if require_a100_80gb is None
        else require_a100_80gb
    )
    if not isinstance(require_a100, bool):
        raise SlurmRenderError("require_a100_80gb must be boolean")

    repo_root = depot / config.storage.source_checkout_relative
    conda_prefix = depot / config.storage.conda_prefix_relative
    scratch_runs = scratch / config.storage.scratch_runs_relative
    slurm_logs = scratch_runs / "slurm"
    directives = [
        "#!/usr/bin/env bash",
        "# Generated from configs/cluster_pilot_n50.v1.json; do not hand-edit.",
        "#SBATCH --job-name=alexdoor-pilot-n50",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-1%{concurrency}",
        f"#SBATCH --gpus-per-node={config.slurm.gpus_per_node}",
        f"#SBATCH --cpus-per-task={selected_cpus}",
        f"#SBATCH --mem={selected_memory}",
        f"#SBATCH --time={selected_wall_time}",
    ]
    if qos is not None:
        directives.append(f"#SBATCH --qos={qos}")
    directives.extend(
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
            f"PARTITION={shlex.quote(partition)}",
            'ATTEMPT_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"',
            'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"',
            'MANIFEST="$REPO_ROOT/outputs/cluster_pilot_n50/pilot_transfer_manifest.json"',
            "",
            '[[ "$ATTEMPT_ID" =~ ^[0-9]+$ ]] || { echo "invalid Slurm array job ID: $ATTEMPT_ID" >&2; exit 19; }',
            '[[ "$TASK_ID" =~ ^[0-9]+$ ]] || { echo "invalid Slurm array task ID: $TASK_ID" >&2; exit 19; }',
            '[[ -d "$DEPOT_ROOT" ]] || { echo "missing depot root: $DEPOT_ROOT" >&2; exit 20; }',
            '[[ -d "$SCRATCH_ROOT" ]] || { echo "missing scratch root: $SCRATCH_ROOT" >&2; exit 21; }',
            '[[ -d "$REPO_ROOT/.git" ]] || { echo "missing source checkout: $REPO_ROOT" >&2; exit 22; }',
            '[[ -x "$CONDA_PREFIX/bin/python" ]] || { echo "missing pilot Conda env: $CONDA_PREFIX" >&2; exit 23; }',
            'export PATH="$CONDA_PREFIX/bin:$PATH"',
            '[[ -f "$MANIFEST" ]] || { echo "missing pilot transfer manifest: $MANIFEST" >&2; exit 24; }',
            'mkdir -p "$SCRATCH_RUNS_ROOT" "$DURABLE_RESULTS_ROOT"',
            'cd "$REPO_ROOT"',
            '[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 25; }',
            '[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 26; }',
            "",
            "case \"$TASK_ID\" in",
            *_render_cells(config.cells),
            '  *) echo "unexpected array task: $TASK_ID" >&2; exit 27 ;;',
            "esac",
            "",
            'CELL_ROOT="$SCRATCH_RUNS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"',
            'CELL_RUNTIME="$CELL_ROOT/runtime"',
            'RUN_OUTPUT_ROOT="$CELL_ROOT/output"',
            'RUN_DIR="$RUN_OUTPUT_ROOT/$EXPERIMENT/$RUN_ID"',
            'PUBLISH_FINAL="$DURABLE_RESULTS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID/$RUN_ID"',
            '[[ ! -e "$CELL_ROOT" ]] || { echo "scratch attempt already exists: $CELL_ROOT" >&2; exit 28; }',
            '[[ ! -e "$PUBLISH_FINAL" ]] || { echo "durable attempt already exists: $PUBLISH_FINAL" >&2; exit 29; }',
            'mkdir -p "$CELL_RUNTIME/environment" "$CELL_RUNTIME/slurm" "$CELL_RUNTIME/status" "$CELL_RUNTIME/wandb"',
            "",
            "write_status() {",
            "  local code=$1",
            "  local status_name status_value",
            "  if [[ $code -eq 0 ]]; then",
            '    status_name="completion.json"',
            '    status_value="COMPLETED"',
            "  else",
            '    status_name="failure.json"',
            '    status_value="FAILED"',
            "  fi",
            '  rm -f "$CELL_RUNTIME/status/completion.json" "$CELL_RUNTIME/status/failure.json"',
            '  local status_tmp="$CELL_RUNTIME/status/.${status_name}.tmp"',
            "  printf '{\"schema\":\"alexdoor_xas.cluster_pilot_cell_status.v2\",\"status\":\"%s\",\"run_id\":\"%s\",\"policy\":\"%s\",\"space\":\"%s\",\"exit_code\":%d,\"source_git_commit\":\"%s\",\"attempt\":{\"slurm_array_job_id\":\"%s\",\"slurm_array_task_id\":\"%s\",\"run_id\":\"%s\"}}\\n' \\",
            '    "$status_value" "$RUN_ID" "$POLICY" "$SPACE" "$code" "$SOURCE_COMMIT" "$ATTEMPT_ID" "$TASK_ID" "$RUN_ID" > "$status_tmp"',
            '  mv "$status_tmp" "$CELL_RUNTIME/status/$status_name"',
            "}",
            "",
            "publish_result() {",
            "  local original_code=$?",
            "  trap - EXIT",
            "  set +e",
            "  local final_code=$original_code",
            "  local status_ok=1",
            '  if ! write_status "$original_code"; then final_code=90; status_ok=0; fi',
            '  local publish_parent="$DURABLE_RESULTS_ROOT/attempts/$ATTEMPT_ID/$TASK_ID"',
            '  local publish_tmp="$publish_parent/.${RUN_ID}.${SLURM_JOB_ID:-$ATTEMPT_ID.$TASK_ID}.tmp"',
            '  local publish_claim="$publish_parent/.${RUN_ID}.claim"',
            '  if [[ $status_ok -eq 0 ]]; then',
            '    echo "could not publish atomic cell status" >&2',
            '  elif ! mkdir -p "$publish_parent"; then',
            '    echo "could not create durable attempt parent: $publish_parent" >&2',
            '    final_code=91',
            '  elif [[ -e "$publish_tmp" || -e "$PUBLISH_FINAL" || -e "$publish_claim" ]]; then',
            '    echo "durable result destination already exists: $PUBLISH_FINAL" >&2',
            "    final_code=91",
            '  elif ! mkdir "$publish_claim"; then',
            '    echo "could not claim durable result destination: $PUBLISH_FINAL" >&2',
            '    final_code=91',
            "  else",
            "    local publish_ok=1",
            '    if ! mkdir -p "$publish_tmp"; then publish_ok=0; fi',
            '    if [[ $publish_ok -eq 1 && -d "$RUN_DIR" ]]; then',
            '      if ! cp -a "$RUN_DIR"/. "$publish_tmp"/; then publish_ok=0; fi',
            "    fi",
            '    if [[ $publish_ok -eq 1 ]]; then',
            '      for name in environment slurm status wandb; do',
            '        if [[ -d "$CELL_RUNTIME/$name" ]] && ! cp -a "$CELL_RUNTIME/$name" "$publish_tmp/$name"; then publish_ok=0; break; fi',
            "      done",
            "    fi",
            '    if [[ $publish_ok -eq 1 && -e "$PUBLISH_FINAL" ]]; then publish_ok=0; fi',
            '    if [[ $publish_ok -eq 1 ]] && ! mv -T "$publish_tmp" "$PUBLISH_FINAL"; then publish_ok=0; fi',
            '    if [[ $publish_ok -eq 0 ]]; then final_code=92; fi',
            '    rmdir "$publish_claim" || true',
            "  fi",
            '  if [[ $final_code -ne $original_code ]]; then write_status "$final_code" || true; fi',
            "  exit \"$final_code\"",
            "}",
            "trap publish_result EXIT",
            "",
            'COMMON_OVERRIDES=(',
            f'  "dataset.task={config.source_dataset.task}"',
            f'  "dataset.version={config.source_dataset.version}"',
            f'  "dataset.obs_preset={config.source_dataset.obs_preset}"',
            f'  "train.seed={config.training.seed}"',
            f'  "train.device={config.training.device}"',
            f'  "train.epochs={config.training.epochs}"',
            f'  "train.val_every={config.training.val_every}"',
            f'  "train.overfit_episodes={config.training.overfit_episodes}"',
            '  "run.run_id=$RUN_ID"',
            '  "run.output_root=$RUN_OUTPUT_ROOT"',
            f'  "+wandb.mode={config.training.wandb_mode}"',
            '  "+wandb.dir=$CELL_RUNTIME/wandb"',
            ")",
            "",
            '"$CONDA_PREFIX/bin/python" scripts/build_cluster_pilot_manifest.py verify \\',
            '  --config configs/cluster_pilot_n50.v1.json --manifest "$MANIFEST"',
            "",
            'PREFLIGHT_ARGS=(',
            '  --config configs/cluster_pilot_n50.v1.json',
            '  --manifest "$MANIFEST"',
            '  --scratch-output "$CELL_RUNTIME"',
            '  --report "$CELL_RUNTIME/environment/preflight_report.json"',
            '  --environment-dir "$CELL_RUNTIME/environment"',
            '  --live-cuda',
            '  --expected-device-count 1',
            '  --requested-partition "$PARTITION"',
            ")",
        ]
    )
    if require_a100:
        directives.append('PREFLIGHT_ARGS+=(--require-a100-80gb)')
    directives.extend(
        [
            "",
            "set +e",
            '"$CONDA_PREFIX/bin/python" scripts/preflight_cluster_pilot.py "${PREFLIGHT_ARGS[@]}" \\',
            '  > "$CELL_RUNTIME/slurm/stdout.log" 2> "$CELL_RUNTIME/slurm/stderr.log"',
            "run_code=$?",
            "if [[ $run_code -eq 0 ]]; then",
            '  "$CONDA_PREFIX/bin/python" "$ENTRYPOINT" "dataset.space=$SPACE" \\',
            '    "${COMMON_OVERRIDES[@]}" "${POLICY_OVERRIDES[@]}" \\',
            '    >> "$CELL_RUNTIME/slurm/stdout.log" 2>> "$CELL_RUNTIME/slurm/stderr.log"',
            "  run_code=$?",
            "fi",
            "set -e",
            'cat "$CELL_RUNTIME/slurm/stdout.log"',
            'cat "$CELL_RUNTIME/slurm/stderr.log" >&2',
            '[[ $run_code -eq 0 ]] || exit "$run_code"',
            "",
            'for required in checkpoints/best.pt checkpoints/last.pt logs/train_log.json \\',
            '  metrics/open_loop.json resolved_config.json; do',
            '  [[ -s "$RUN_DIR/$required" ]] || { echo "missing completed artifact: $RUN_DIR/$required" >&2; exit 93; }',
            "done",
            '[[ -s "$CELL_RUNTIME/environment/environment_inventory.json" ]] || exit 94',
            '[[ -s "$CELL_RUNTIME/environment/requirements.lock" ]] || exit 95',
            '[[ -n "$(find "$CELL_RUNTIME/wandb" -type f -print -quit)" ]] || exit 96',
            "exit 0",
            "",
        ]
    )
    return "\n".join(directives)


def _render_cells(cells: tuple[PilotCell, ...]) -> list[str]:
    lines: list[str] = []
    experiments = {"act": "act_door_push", "diffusion": "diffusion_door_push"}
    for cell in cells:
        lines.extend(
            [
                f"  {cell.index})",
                f"    POLICY={shlex.quote(cell.policy)}",
                f"    SPACE={shlex.quote(cell.space)}",
                f"    RUN_ID={shlex.quote(cell.run_id)}",
                f"    ENTRYPOINT={shlex.quote(cell.entrypoint)}",
                f"    EXPERIMENT={shlex.quote(experiments[cell.policy])}",
                "    POLICY_OVERRIDES=(",
            ]
        )
        for key, value in cell.overrides.items():
            rendered_value = str(value).lower() if isinstance(value, bool) else str(value)
            lines.append(f"      {shlex.quote(f'{key}={rendered_value}')}")
        lines.extend(["    )", "    ;;"])
    return lines


def _absolute_path(name: str, value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(character in str(path) for character in ("\n", "\r", "\0")):
        raise SlurmRenderError(f"{name} must be a safe absolute path")
    return path


def _directive(name: str, value: str) -> str:
    if not isinstance(value, str) or SAFE_DIRECTIVE_RE.fullmatch(value) is None:
        raise SlurmRenderError(f"{name} is required and contains unsupported characters")
    return value


__all__ = ["SlurmRenderError", "render_slurm_script"]
