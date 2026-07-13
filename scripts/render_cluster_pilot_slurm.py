#!/usr/bin/env python
"""Render, but never submit, the deterministic Gilbreth pilot Slurm array."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.slurm import SlurmRenderError, render_slurm_script


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cluster_pilot_n50.v1.json")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depot-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-results-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--memory", default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--wall-time", default=None)
    parser.add_argument("--require-a100-80gb", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_pilot_config(paths.REPO_ROOT / args.config)
        manifest = json.loads((paths.REPO_ROOT / args.manifest).resolve().read_text())
        source_commit = manifest["source_git"]["commit"]
        rendered = render_slurm_script(
            config,
            source_commit=source_commit,
            depot_root=args.depot_root,
            scratch_root=args.scratch_root,
            durable_results_root=args.durable_results_root,
            account=args.account,
            partition=args.partition,
            qos=args.qos,
            memory=args.memory,
            cpus_per_task=args.cpus_per_task,
            wall_time=args.wall_time,
            require_a100_80gb=args.require_a100_80gb,
        )
        output = (paths.REPO_ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(rendered)
            temporary.chmod(0o750)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        print("PASS: rendered two-cell Slurm pilot array (not submitted)")
        print(f"output: {output}")
        print(f"source commit: {source_commit}")
        return 0
    except (OSError, KeyError, json.JSONDecodeError, PilotConfigError, SlurmRenderError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
