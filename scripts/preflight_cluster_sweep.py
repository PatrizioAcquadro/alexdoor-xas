#!/usr/bin/env python
"""Run pure and optional live-CUDA preflight for one full-sweep cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.preflight import (
    dependency_inventory,
    write_environment_reports,
)
from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.cluster_sweep.preflight import (
    atomic_json,
    probe_cuda_device,
    run_sweep_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--live-cuda", action="store_true")
    parser.add_argument("--expected-device-count", type=int, default=1)
    parser.add_argument("--requested-partition")
    parser.add_argument("--require-a100-80gb", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_sweep_config(args.config)
        manifest = json.loads(args.manifest.read_text())
        dependencies = dependency_inventory()
        report = run_sweep_preflight(
            repo_root=paths.REPO_ROOT,
            config=config,
            manifest=manifest,
            scratch_output=args.scratch_output,
            dependency_probe=lambda: dependencies,
        )
        cuda_report = None
        if args.live_cuda:
            import torch

            cuda_report = probe_cuda_device(
                torch,
                expected_device_count=args.expected_device_count,
                require_a100_80gb=args.require_a100_80gb,
            )
            report["cuda_probe"] = cuda_report
            report["requested_partition"] = args.requested_partition
        write_environment_reports(
            args.environment_dir, dependencies=dependencies, cuda_report=cuda_report
        )
        atomic_json(args.report, report)
        print("PASS: cluster sweep preflight verified")
    except Exception as error:  # noqa: BLE001 - CLI reports the complete gate.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
