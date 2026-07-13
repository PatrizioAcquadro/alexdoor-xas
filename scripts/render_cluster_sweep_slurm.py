#!/usr/bin/env python
"""Render, but never submit, the stable 16-cell full-sweep Slurm array."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.cluster_sweep.slurm import render_sweep_slurm_script


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depot-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-results-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--qos")
    parser.add_argument("--memory")
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--wall-time")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--require-a100-80gb", action="store_true", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        from dataclasses import replace

        config = load_sweep_config(args.config)
        if args.concurrency is not None:
            config = replace(
                config,
                slurm=replace(config.slurm, array_max_concurrent=args.concurrency),
            )
        manifest = json.loads(args.manifest.read_text())
        source_commit = manifest["source_git"]["commit"]
        rendered = render_sweep_slurm_script(
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"PASS: rendered 16-cell Slurm array: {args.output}")
    except Exception as error:  # noqa: BLE001 - CLI reports every render gate.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
