#!/usr/bin/env python
"""Build or verify the clean-tree full-sweep transfer manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.cluster_sweep.transfer import (
    build_sweep_transfer_manifest,
    verify_sweep_transfer_manifest,
    write_sweep_transfer_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--config", type=Path, default=Path("configs/cluster_sweep.v1.json"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cluster_sweep"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_sweep_config(args.config)
        if args.command == "build":
            manifest = build_sweep_transfer_manifest(paths.REPO_ROOT, config)
            manifest_path, file_list, command = write_sweep_transfer_artifacts(
                paths.REPO_ROOT, manifest, output_dir=args.output_dir
            )
            print(
                f"PASS: built {manifest_path} ({manifest['file_count']} files, "
                f"{manifest['total_size_bytes']} bytes); list={file_list}; command={command}"
            )
        else:
            if args.manifest is None:
                raise ValueError("verify requires --manifest")
            manifest = json.loads(args.manifest.read_text())
            failures = verify_sweep_transfer_manifest(manifest, paths.REPO_ROOT, config)
            if failures:
                raise ValueError("; ".join(failures))
            print("PASS: sweep transfer manifest verified")
    except Exception as error:  # noqa: BLE001 - CLI reports all fail-closed gates.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
