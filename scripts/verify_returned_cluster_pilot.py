#!/usr/bin/env python
"""Verify returned hashes and load both pilot checkpoints without Isaac."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.returns import (
    ReturnManifestError,
    verify_return_checkpoints,
    verify_return_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--attempt-id",
        required=True,
        help="Explicit SLURM_ARRAY_JOB_ID selecting one returned attempt.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cluster_pilot_n50.v1.json")
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_pilot_config(paths.REPO_ROOT / args.config)
        manifest = json.loads(args.manifest.resolve().read_text())
        failures = verify_return_manifest(
            manifest,
            args.results_root,
            config,
            attempt_id=args.attempt_id,
        )
        if failures:
            raise ReturnManifestError("; ".join(failures))
        checkout_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if checkout_commit != manifest.get("source_git_commit"):
            raise ReturnManifestError(
                "Ubuntu checkout commit does not match the returned pilot source commit"
            )
        statuses = verify_return_checkpoints(
            args.results_root,
            config,
            attempt_id=args.attempt_id,
        )
        print("PASS: returned pilot hashes and CPU checkpoint loads verified")
        for run_id, status in statuses.items():
            print(f"- {run_id}: {status}")
        return 0
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        PilotConfigError,
        ReturnManifestError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
