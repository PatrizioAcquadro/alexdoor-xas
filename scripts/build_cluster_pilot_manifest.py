#!/usr/bin/env python
"""Build or verify the exact local N50 Gilbreth pilot transfer package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_pilot.config import PilotConfigError, load_pilot_config
from alexdoor_xas.cluster_pilot.transfer import (
    DEFAULT_OUTPUT_DIR,
    PilotTransferError,
    build_transfer_manifest,
    pilot_rsync_template,
    verify_transfer_manifest,
    write_transfer_artifacts,
)

DEFAULT_CONFIG = Path("configs/cluster_pilot_n50.v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build from a clean committed checkout.")
    verify = subparsers.add_parser("verify", help="Verify a transferred manifest in place.")
    for command in (build, verify):
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_pilot_config(paths.REPO_ROOT / args.config)
        if args.command == "build":
            manifest = build_transfer_manifest(paths.REPO_ROOT, config)
            manifest_path, files_path, command_path = write_transfer_artifacts(
                paths.REPO_ROOT,
                manifest,
                output_dir=args.output_dir,
            )
            print("PASS: Gilbreth pilot transfer package built")
            print(f"manifest: {manifest_path}")
            print(f"files: {files_path}")
            print(f"rsync template: {command_path}")
            print(f"file count: {manifest['file_count']}")
            print(f"total bytes: {manifest['total_size_bytes']}")
            print(f"source commit: {manifest['source_git']['commit']}")
            print(pilot_rsync_template())
            return 0

        manifest_path = (paths.REPO_ROOT / args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        failures = verify_transfer_manifest(
            manifest,
            paths.REPO_ROOT,
            config,
            require_tracked=True,
        )
        if failures:
            print("FAIL: Gilbreth pilot transfer verification")
            for failure in failures:
                print(f"- {failure}")
            return 1
        print("PASS: Gilbreth pilot transfer manifest verified")
        print(f"source commit: {manifest['source_git']['commit']}")
        print(f"file count: {manifest['file_count']}")
        return 0
    except (OSError, json.JSONDecodeError, PilotConfigError, PilotTransferError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
