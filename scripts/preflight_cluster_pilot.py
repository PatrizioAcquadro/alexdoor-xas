#!/usr/bin/env python
"""Run the pure and optional live-CUDA Gilbreth pilot preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.preflight import (
    ClusterPreflightError,
    dependency_inventory,
    probe_cuda_device,
    run_pure_preflight,
    write_environment_reports,
)
from alexdoor_xas.cluster_pilot.transfer import PilotTransferError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cluster_pilot_n50.v1.json")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, required=True)
    parser.add_argument("--live-cuda", action="store_true")
    parser.add_argument("--expected-device-count", type=int, default=1)
    parser.add_argument("--requested-partition", type=str, default=None)
    parser.add_argument("--require-a100-80gb", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_pilot_config(paths.REPO_ROOT / args.config)
        manifest = json.loads((paths.REPO_ROOT / args.manifest).resolve().read_text())
        dependencies = dependency_inventory()
        report = run_pure_preflight(
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
            args.environment_dir,
            dependencies=dependencies,
            cuda_report=cuda_report,
        )
        _atomic_write(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("PASS: pure non-Isaac Gilbreth pilot preflight")
        print(f"source commit: {report['source_git_commit']}")
        print(f"CUDA: {'PASS' if cuda_report is not None else 'NOT_RUN'}")
        return 0
    except (
        OSError,
        json.JSONDecodeError,
        PilotConfigError,
        PilotTransferError,
        ClusterPreflightError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
