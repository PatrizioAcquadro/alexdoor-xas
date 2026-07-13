#!/usr/bin/env python
"""Build the exact-attempt durable return package for all 16 sweep cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.cluster_sweep.returns import (
    build_sweep_return_manifest,
    write_return_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_sweep_config(args.config)
        transfer = json.loads(args.manifest.read_text())
        manifest = build_sweep_return_manifest(
            args.results_root,
            attempt_id=args.attempt_id,
            config=config,
            transfer_manifest=transfer,
        )
        manifest_path, file_list, command = write_return_artifacts(
            args.results_root, manifest
        )
        print(
            f"PASS: built 16-cell return manifest {manifest_path} "
            f"({manifest['file_count']} files); list={file_list}; command={command}"
        )
    except Exception as error:  # noqa: BLE001 - CLI reports exact return gate.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
