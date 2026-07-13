#!/usr/bin/env python
"""Verify returned sweep hashes and CPU-load all 16 best checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas.cluster_sweep.config import load_sweep_config
from alexdoor_xas.cluster_sweep.returns import (
    verify_return_control_files,
    verify_sweep_checkpoints,
    verify_sweep_return_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--return-manifest", type=Path, required=True)
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_sweep_config(args.config)
        returned = json.loads(args.return_manifest.read_text())
        transfer = json.loads(args.transfer_manifest.read_text())
        failures = verify_sweep_return_manifest(
            returned,
            args.results_root,
            attempt_id=args.attempt_id,
            config=config,
            transfer_manifest=transfer,
        )
        controls = verify_return_control_files(
            args.results_root,
            returned,
            manifest_path=args.return_manifest,
            files_path=args.return_manifest.with_name("return-files.txt"),
        )
        failures.extend(controls)
        if failures:
            raise ValueError("; ".join(failures))
        statuses = verify_sweep_checkpoints(
            args.results_root,
            attempt_id=args.attempt_id,
            config=config,
            transfer_manifest=transfer,
        )
        if len(statuses) != 16 or set(statuses.values()) != {"CPU_LOAD_PASS"}:
            raise ValueError("not all 16 best checkpoints passed CPU loading")
        print("PASS: returned sweep hashes and all 16 CPU checkpoint loads verified")
    except Exception as error:  # noqa: BLE001 - CLI reports exact verification gate.
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
