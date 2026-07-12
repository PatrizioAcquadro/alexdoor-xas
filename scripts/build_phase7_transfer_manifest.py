#!/usr/bin/env python
"""Generate and verify the complete Phase 6 -> Phase 7 transfer inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

REQUIRED_CATEGORIES = (
    "dataset",
    "split",
    "norm_stats",
    "checkpoint",
    "evaluation",
    "summary",
    "report",
)
RUN_DIRS = (
    "outputs/door_push_alex_v2/act_door_push/local_smoke_act_a2_n50_seed0",
    "outputs/door_push_alex_v2/act_door_push/local_smoke_act_a3_n50_seed0",
    "outputs/door_push_alex_v2/diffusion_door_push/local_smoke_diffusion_a2_n50_seed0",
    "outputs/door_push_alex_v2/diffusion_door_push/local_smoke_diffusion_a3_n50_seed0",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_transfer_manifest(
    repo_root: Path,
    artifacts: dict[str, list[Path]],
    *,
    phase7_authorized: bool,
) -> dict[str, Any]:
    """Hash every required artifact and self-verify the resulting inventory."""
    repo_root = repo_root.resolve()
    missing_categories = [name for name in REQUIRED_CATEGORIES if not artifacts.get(name)]
    if missing_categories:
        raise ValueError(f"required artifact categories are empty: {missing_categories}")
    entries = []
    seen: set[str] = set()
    for category in REQUIRED_CATEGORIES:
        for path in sorted({item.resolve() for item in artifacts[category]}):
            try:
                relative = path.relative_to(repo_root).as_posix()
            except ValueError as error:
                raise ValueError(f"artifact is outside the repository: {path}") from error
            if relative in seen:
                raise ValueError(f"artifact appears in multiple categories: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(relative)
            entries.append(
                {
                    "category": category,
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema": "alexdoor_xas.phase7_transfer_manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase7_authorized": bool(phase7_authorized),
        "files": entries,
        "category_counts": {
            category: sum(entry["category"] == category for entry in entries)
            for category in REQUIRED_CATEGORIES
        },
        "verification": {"algorithm": "sha256", "status": "PENDING", "failures": []},
    }
    failures = verify_transfer_manifest(manifest, repo_root)
    manifest["verification"] = {
        "algorithm": "sha256",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    return manifest


def verify_transfer_manifest(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    """Return every missing, size-changed, or hash-changed manifest entry."""
    failures: list[str] = []
    root = repo_root.resolve()
    for entry in manifest.get("files", []):
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes repository: {entry['path']}")
            continue
        if not path.is_file():
            failures.append(f"missing file: {entry['path']}")
            continue
        if path.stat().st_size != entry["size_bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['path']}")
    return failures


def render_phase6_report(summary_path: Path, report_path: Path) -> Path:
    """Render the final report from summary and evaluation JSONs only."""
    summary = json.loads(summary_path.read_text())
    authorized = all(
        summary.get(field) == "PASS"
        for field in ("metadata_coverage", "protocol_consistency", "safety_readiness")
    )
    lines = [
        "# Phase 6 closeout report",
        "",
        f"- Metadata coverage: {summary.get('metadata_coverage')}",
        f"- Protocol consistency: {summary.get('protocol_consistency')}",
        f"- Safety readiness: {summary.get('safety_readiness')}",
        f"- Phase 7 authorized: {'YES' if authorized else 'NO'}",
        "",
        "## Smoke matrix",
        "",
        "| Run | Success | Peak force | Force exceedances | Readiness |",
        "|---|---:|---:|---:|---|",
    ]
    force_rows: list[tuple[str, dict[str, Any]]] = []
    for name, run in summary.get("runs", {}).items():
        overall = run["overall"]
        lines.append(
            f"| {name} | {overall['n_success']}/{overall['n_rollouts']} | "
            f"{overall['peak_force_n']:.1f} N | "
            f"{overall['n_force_exceeds_admission_bound']} | "
            f"{run['safety_readiness']['status']} |"
        )
        for pose in run.get("poses", {}).values():
            payload = json.loads(Path(pose["eval_json"]).read_text())
            force_rows.extend(
                (name, row)
                for row in payload.get("rollouts", [])
                if row.get("force_exceeds_admission_bound")
            )
    lines.extend(["", "## Force adjudication", ""])
    if not force_rows:
        lines.append("No rollout exceeded the unchanged 200 N bound.")
    for name, row in force_rows:
        evidence = row.get("force_trace_evidence") or {}
        lines.append(
            f"- `{name}` seed {row.get('seed')}: {evidence.get('peak_force_n'):.1f} N at "
            f"tick {evidence.get('peak_tick')}; {evidence.get('n_exceedance_ticks')} "
            f"over-bound tick(s), contact={evidence.get('peak_contact')}, "
            f"adapter={evidence.get('peak_status')}, trace `{evidence.get('trace_sha256')}`."
        )
    if force_rows:
        lines.extend(
            [
                "",
                "The trace evidence localizes the exceedances but does not prove that forces "
                "above the 200 N safety bound are acceptable for transfer. The bound remains "
                "unchanged, the warnings remain visible, and Phase 7 is blocked pending a "
                "safety-resolving policy or controller change followed by regeneration.",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def canonical_artifacts(
    repo_root: Path, summary_path: Path, report_path: Path
) -> dict[str, list[Path]]:
    """Collect the exact ignored artifact set required for Phase 7 transfer."""
    dataset_root = repo_root / "datasets" / "door_push_alex_v2"
    dataset_files = []
    norm_stats = []
    for space in (
        "A1_joint_delta",
        "A2_ee_delta",
        "A3_obj_rel_ee_delta",
        "A4_obj_centric_chunk",
    ):
        for path in (dataset_root / space / "v2_pose").glob("*"):
            if path.is_file() and path.name == "norm_stats.json":
                norm_stats.append(path)
            elif path.is_file():
                dataset_files.append(path)
    run_dirs = [repo_root / relative for relative in RUN_DIRS]
    return {
        "dataset": dataset_files,
        "split": [dataset_root / "splits" / "v2_pose.json"],
        "norm_stats": norm_stats,
        "checkpoint": [run_dir / "checkpoints" / "best.pt" for run_dir in run_dirs],
        "evaluation": [
            path
            for run_dir in run_dirs
            for path in sorted((run_dir / "metrics").glob("*_eval_D[0-4].json"))
        ],
        "summary": [summary_path],
        "report": [report_path],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/local_smoke_n50/summary.json")
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/local_smoke_n50/report.md"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/local_smoke_n50/phase7_transfer_manifest.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repo_root = paths.REPO_ROOT.resolve()
    if args.verify:
        manifest = json.loads(args.out.read_text())
        failures = verify_transfer_manifest(manifest, repo_root)
        print("PASS" if not failures else "FAIL: " + "; ".join(failures))
        return 0 if not failures else 1
    report = render_phase6_report(args.summary, args.report)
    summary = json.loads(args.summary.read_text())
    authorized = all(
        summary.get(field) == "PASS"
        for field in ("metadata_coverage", "protocol_consistency", "safety_readiness")
    )
    manifest = build_transfer_manifest(
        repo_root,
        canonical_artifacts(repo_root, args.summary, report),
        phase7_authorized=authorized,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"PASS: hashed {len(manifest['files'])} artifacts; "
        f"Phase 7 authorized={manifest['phase7_authorized']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
