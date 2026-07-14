#!/usr/bin/env python
"""Prepare, run, audit, and report the frozen Phase 3 unified evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alexdoor_xas.eval.phase3_unified import (
    UnifiedEvalError,
    audit_evidence,
    load_plan,
    prepare_workspace,
    run_cell,
    run_preflight,
    verify_immutable_inventory,
    write_report_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/phase3_unified_eval.v1.json"),
        help="Frozen evaluation plan.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Validate artifacts and resolve the 16-cell plan.")
    preflight = subparsers.add_parser("preflight", help="Run the bounded two-policy preflight.")
    preflight.add_argument("--run-id", default=None, help="Run only one frozen preflight cell.")
    run = subparsers.add_parser("run-cell", help="Run/resume one 36-rollout primary cell.")
    run.add_argument("--run-id", required=True)
    subparsers.add_parser("audit", help="Audit all cells without producing comparisons.")
    subparsers.add_parser("report", help="Normalize complete cells and write curated outputs.")
    subparsers.add_parser("verify-immutable", help="Recheck every returned path, size, and hash.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = load_plan(args.plan)
        if args.command == "prepare":
            resolved = prepare_workspace(plan)
            print(
                f"PASS: prepared {len(resolved['cells'])} cells; "
                f"resolved plan {plan.workspace_root / 'provenance/evaluation_plan.resolved.json'}"
            )
        elif args.command == "preflight":
            outputs = run_preflight(plan, args.run_id)
            print("PASS: preflight artifacts")
            for path in outputs:
                print(path)
        elif args.command == "run-cell":
            print(f"PASS: cell completion {run_cell(plan, args.run_id)}")
        elif args.command == "audit":
            audit = audit_evidence(plan)
            audit.pop("_rows")
            print(json.dumps(audit, indent=2, sort_keys=True))
        elif args.command == "report":
            outputs = write_report_artifacts(plan)
            print("PASS: curated Phase 3 artifacts")
            for label, path in outputs.items():
                print(f"{label}: {path}")
        elif args.command == "verify-immutable":
            failures = verify_immutable_inventory(plan)
            if failures:
                raise UnifiedEvalError("; ".join(failures))
            print("PASS: returned package exact paths, sizes, and hashes are unchanged")
        else:  # pragma: no cover - argparse prevents this branch.
            raise UnifiedEvalError(f"unknown command {args.command!r}")
    except (OSError, ValueError, KeyError, UnifiedEvalError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
