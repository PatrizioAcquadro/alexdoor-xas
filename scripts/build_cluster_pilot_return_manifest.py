#!/usr/bin/env python
"""Build or verify the durable Gilbreth-to-Ubuntu pilot return package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.returns import (
    ReturnManifestError,
    build_return_manifest,
    return_rsync_template,
    verify_return_manifest,
    write_return_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--results-root", type=Path, required=True)
        command.add_argument(
            "--config", type=Path, default=Path("configs/cluster_pilot_n50.v1.json")
        )
        command.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_pilot_config(paths.REPO_ROOT / args.config)
        source_or_return = json.loads(args.manifest.resolve().read_text())
        if args.command == "build":
            manifest = build_return_manifest(args.results_root, config, source_or_return)
            manifest_path, files_path, command_path = write_return_artifacts(
                args.results_root, manifest
            )
            print("PASS: durable pilot return package built")
            print(f"manifest: {manifest_path}")
            print(f"files: {files_path}")
            print(f"rsync template: {command_path}")
            print(return_rsync_template())
            return 0
        failures = verify_return_manifest(source_or_return, args.results_root, config)
        if failures:
            print("FAIL: returned pilot manifest verification")
            for failure in failures:
                print(f"- {failure}")
            return 1
        print("PASS: returned pilot manifest verified")
        return 0
    except (
        OSError,
        json.JSONDecodeError,
        PilotConfigError,
        ReturnManifestError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
