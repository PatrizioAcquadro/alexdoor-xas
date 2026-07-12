#!/usr/bin/env python
"""Generate and fail-closed verify the local-to-cluster transfer inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

SCHEMA = "alexdoor_xas.cluster_transfer_manifest.v1"
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
DATASET_SPACES = (
    "A1_joint_delta",
    "A2_ee_delta",
    "A3_obj_rel_ee_delta",
)
READINESS_FIELDS = ("metadata_coverage", "protocol_consistency", "safety_readiness")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return the exact commit/ref and clean-tree state used for authorization."""
    commit_result = _git(repo_root, "rev-parse", "HEAD")
    if commit_result.returncode:
        raise RuntimeError(f"cannot read source Git commit: {commit_result.stderr.strip()}")
    branch_result = _git(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    status_result = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status_result.returncode:
        raise RuntimeError(f"cannot read source Git status: {status_result.stderr.strip()}")
    return {
        "commit": commit_result.stdout.strip(),
        "branch": branch,
        "detached": branch is None,
        "clean_tree": not bool(status_result.stdout.strip()),
    }


def _relative(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the repository: {path}") from error


def _inventory_map(
    repo_root: Path, artifacts: dict[str, list[Path]]
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for category, category_paths in artifacts.items():
        for path in category_paths:
            relative = _relative(repo_root, path)
            previous = inventory.get(relative)
            if previous is not None:
                raise ValueError(
                    f"artifact appears in multiple categories: {relative} "
                    f"({previous}, {category})"
                )
            inventory[relative] = category
    return inventory


def _readiness_passes(statuses: dict[str, Any]) -> bool:
    return all(statuses.get(field) == "PASS" for field in READINESS_FIELDS)


def build_transfer_manifest(
    repo_root: Path,
    artifacts: dict[str, list[Path]],
    *,
    readiness_statuses: dict[str, Any],
    regeneration: dict[str, Any],
    expected_artifacts: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    """Hash the required inventory and derive authorization from verified state."""
    repo_root = repo_root.resolve()
    missing_categories = [name for name in REQUIRED_CATEGORIES if not artifacts.get(name)]
    if missing_categories:
        raise ValueError(f"required artifact categories are empty: {missing_categories}")
    unexpected = sorted(set(artifacts) - set(REQUIRED_CATEGORIES))
    if unexpected:
        raise ValueError(f"unexpected artifact categories: {unexpected}")
    _inventory_map(repo_root, artifacts)

    entries: list[dict[str, Any]] = []
    for category in REQUIRED_CATEGORIES:
        for path in sorted((item.resolve() for item in artifacts[category]), key=str):
            relative = _relative(repo_root, path)
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append(
                {
                    "category": category,
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_git": git_state(repo_root),
        "readiness": {field: readiness_statuses.get(field) for field in READINESS_FIELDS},
        "regeneration": regeneration,
        "files": entries,
        "category_counts": dict(Counter(entry["category"] for entry in entries)),
        "verification": {"algorithm": "sha256", "status": "PENDING", "failures": []},
        "cluster_sweep_authorized": None,
    }
    failures = verify_transfer_manifest(
        manifest,
        repo_root,
        expected_artifacts=expected_artifacts or artifacts,
        check_declared_outcome=False,
    )
    manifest["verification"] = {
        "algorithm": "sha256",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    manifest["cluster_sweep_authorized"] = bool(
        not failures and _readiness_passes(manifest["readiness"])
    )
    return manifest


def verify_transfer_manifest(
    manifest: dict[str, Any],
    repo_root: Path,
    *,
    expected_artifacts: dict[str, list[Path]] | None = None,
    check_declared_outcome: bool = True,
) -> list[str]:
    """Return all schema, inventory, content, and source-checkout failures."""
    failures: list[str] = []
    root = repo_root.resolve()
    if manifest.get("schema") != SCHEMA:
        failures.append(f"schema must be {SCHEMA!r}, got {manifest.get('schema')!r}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        failures.append("files array is missing or empty")
        files = []

    actual_counts: Counter[str] = Counter()
    categories: set[str] = set()
    path_categories: dict[str, set[str]] = {}
    actual_inventory: dict[str, str] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is not an object")
            continue
        category = entry.get("category")
        relative = entry.get("path")
        if not isinstance(category, str):
            failures.append(f"files[{index}] has invalid category")
            continue
        categories.add(category)
        actual_counts[category] += 1
        if not isinstance(relative, str) or not relative:
            failures.append(f"files[{index}] has invalid path")
            continue
        path_categories.setdefault(relative, set()).add(category)
        if relative in actual_inventory:
            failures.append(f"duplicate path: {relative}")
        else:
            actual_inventory[relative] = category

        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes repository: {relative}")
            continue
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
            failures.append(f"malformed sha256: {relative}")
        size_bytes = entry.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            failures.append(f"invalid size_bytes: {relative}")
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        if isinstance(size_bytes, int) and path.stat().st_size != size_bytes:
            failures.append(f"size mismatch: {relative}")
        if isinstance(sha256, str) and SHA256_RE.fullmatch(sha256) and _sha256(path) != sha256:
            failures.append(f"sha256 mismatch: {relative}")

    required = set(REQUIRED_CATEGORIES)
    missing_categories = sorted(required - categories)
    unexpected_categories = sorted(categories - required)
    if missing_categories:
        failures.append(f"missing required categories: {missing_categories}")
    if unexpected_categories:
        failures.append(f"unexpected categories: {unexpected_categories}")
    for relative, assigned in path_categories.items():
        if len(assigned) > 1:
            failures.append(
                f"path assigned to multiple categories: {relative} -> {sorted(assigned)}"
            )

    declared_counts = manifest.get("category_counts")
    expected_counts = {category: actual_counts.get(category, 0) for category in REQUIRED_CATEGORIES}
    if declared_counts != expected_counts:
        failures.append(
            f"category_counts inconsistent: declared={declared_counts!r}, "
            f"actual={expected_counts!r}"
        )

    if expected_artifacts is None:
        expected_artifacts = canonical_artifacts(
            root,
            root / "outputs/local_smoke_n50/summary.json",
            root / "outputs/local_smoke_n50/report.md",
        )
    try:
        expected_inventory = _inventory_map(root, expected_artifacts)
    except ValueError as error:
        failures.append(f"invalid canonical artifact inventory: {error}")
        expected_inventory = {}
    missing_inventory = sorted(set(expected_inventory.items()) - set(actual_inventory.items()))
    unexpected_inventory = sorted(set(actual_inventory.items()) - set(expected_inventory.items()))
    if missing_inventory or unexpected_inventory:
        failures.append(
            "canonical artifact inventory mismatch: "
            f"missing={missing_inventory}, unexpected={unexpected_inventory}"
        )

    source = manifest.get("source_git")
    try:
        current = git_state(root)
    except RuntimeError as error:
        failures.append(f"source Git state unavailable: {error}")
        current = None
    if not isinstance(source, dict) or not source.get("commit"):
        failures.append("source Git metadata or source commit is missing")
    elif current is not None:
        if source.get("commit") != current["commit"]:
            failures.append(
                f"source commit mismatch: manifest={source.get('commit')}, "
                f"checkout={current['commit']}"
            )
        if (
            source.get("branch") != current["branch"]
            or source.get("detached") != current["detached"]
        ):
            failures.append("source Git branch/detached marker mismatch")
        if source.get("clean_tree") is not True or current["clean_tree"] is not True:
            failures.append("source tree is dirty; authorization requires a clean checkout")

    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict) or any(field not in readiness for field in READINESS_FIELDS):
        failures.append("readiness fields are missing")
        readiness = {}
    if not isinstance(manifest.get("regeneration"), dict) or not manifest.get("regeneration"):
        failures.append("regeneration identifiers are missing")

    if check_declared_outcome:
        expected_status = "PASS" if not failures else "FAIL"
        declared_verification = manifest.get("verification") or {}
        if declared_verification.get("algorithm") != "sha256":
            failures.append("verification algorithm must be sha256")
        if declared_verification.get("status") != expected_status:
            failures.append(
                f"verification status mismatch: declared={declared_verification.get('status')!r}, "
                f"computed={expected_status!r}"
            )
        derived_authorization = bool(not failures and _readiness_passes(readiness))
        if manifest.get("cluster_sweep_authorized") is not derived_authorization:
            failures.append(
                "cluster_sweep_authorized is not the value derived from readiness, "
                "inventory, hashes, and source checkout"
            )
    return failures


def render_local_stabilization_report(summary_path: Path, report_path: Path) -> Path:
    """Render the local closeout report from summary and evaluation JSON only."""
    summary = json.loads(summary_path.read_text())
    eligible = all(summary.get(field) == "PASS" for field in READINESS_FIELDS)
    lines = [
        "# Local post-Phase 3.3 stabilization closeout report",
        "",
        f"- Metadata coverage: {summary.get('metadata_coverage')}",
        f"- Protocol consistency: {summary.get('protocol_consistency')}",
        f"- Safety readiness: {summary.get('safety_readiness')}",
        f"- Local evidence permits cluster sweep: {'YES' if eligible else 'NO'}",
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
            payload = json.loads((paths.REPO_ROOT / pose["eval_json"]).read_text())
            force_rows.extend(
                (name, row)
                for row in payload.get("rollouts", [])
                if row.get("force_exceeds_admission_bound")
            )
    lines.extend(["", "## Force adjudication", ""])
    if not force_rows:
        lines.append("No primary rollout exceeded the unchanged 200 N bound.")
    else:
        for name, row in force_rows:
            evidence = row.get("force_trace_evidence") or {}
            lines.append(
                f"- `{name}` seed {row.get('seed')}: {evidence.get('peak_force_n'):.1f} N at "
                f"tick {evidence.get('peak_tick')}; {evidence.get('n_exceedance_ticks')} "
                f"over-bound tick(s), contact={evidence.get('peak_contact')}, "
                f"adapter={evidence.get('peak_status')}, trace "
                f"`{evidence.get('trace_sha256')}`."
            )
        lines.extend(
            [
                "",
                "Trace localization does not make an over-bound force acceptable. The "
                "200 N bound remains unchanged and the cluster sweep stays blocked until "
                "regenerated primary evidence passes safety readiness.",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def canonical_artifacts(
    repo_root: Path, summary_path: Path, report_path: Path
) -> dict[str, list[Path]]:
    """Reconstruct the exact required artifact set independently of the manifest."""
    dataset_root = repo_root / "datasets" / "door_push_alex_v2"
    split = dataset_root / "splits" / "v2_pose.json"
    episode_ids: list[str] = []
    if split.is_file():
        split_payload = json.loads(split.read_text())
        splits = split_payload.get("splits") or {}
        episode_ids = sorted(
            {str(item) for name in ("train", "val", "test") for item in splits.get(name, [])}
        )
    dataset_files: list[Path] = []
    norm_stats: list[Path] = []
    for space in DATASET_SPACES:
        space_root = dataset_root / space / "v2_pose"
        for episode_id in episode_ids:
            stem = f"episode_{episode_id[:8]}"
            dataset_files.extend([space_root / f"{stem}.hdf5", space_root / f"{stem}.meta.json"])
        dataset_files.extend([space_root / "manifest.json", space_root / "meta.json"])
        norm_stats.append(space_root / "norm_stats.json")
    a4_root = dataset_root / "A4_obj_centric_chunk" / "v2_pose"
    dataset_files.extend(
        [a4_root / "episodes.jsonl", a4_root / "manifest.json", a4_root / "meta.json"]
    )
    run_dirs = [repo_root / relative for relative in RUN_DIRS]
    evaluations = []
    for run_dir in run_dirs:
        policy = "act" if "/act_" in str(run_dir) else "diffusion"
        evaluations.extend(
            run_dir / "metrics" / f"{policy}_eval_D{pose}.json" for pose in range(5)
        )
    return {
        "dataset": dataset_files,
        "split": [split],
        "norm_stats": norm_stats,
        "checkpoint": [run_dir / "checkpoints" / "best.pt" for run_dir in run_dirs],
        "evaluation": evaluations,
        "summary": [summary_path],
        "report": [report_path],
    }


def regeneration_identifiers(repo_root: Path) -> dict[str, Any]:
    """Stable, non-secret identifiers for reproducing the transferred artifacts."""
    eval_plan = repo_root / "configs/local_smoke_eval_plan_n50.json"
    pose_plan = repo_root / "configs/door_pose_plan_v2_pose.json"
    return {
        "dataset_task": "door_push_alex_v2",
        "dataset_version": "v2_pose",
        "dataset_episode_count": 50,
        "run_ids": [Path(item).name for item in RUN_DIRS],
        "primary_pose_ids": [f"D{index}" for index in range(5)],
        "evaluation_plan": {
            "path": _relative(repo_root, eval_plan),
            "sha256": _sha256(eval_plan),
        },
        "door_pose_plan": {
            "path": _relative(repo_root, pose_plan),
            "sha256": _sha256(pose_plan),
        },
        "manifest_builder": "scripts/build_cluster_transfer_manifest.py",
        "summary_builder": "scripts/summarize_smoke_eval.py",
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
        default=Path("outputs/local_smoke_n50/cluster_transfer_manifest.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repo_root = paths.REPO_ROOT.resolve()
    artifacts = canonical_artifacts(repo_root, args.summary, args.report)
    if args.verify:
        manifest = json.loads(args.out.read_text())
        failures = verify_transfer_manifest(
            manifest, repo_root, expected_artifacts=artifacts
        )
        print("PASS" if not failures else "FAIL: " + "; ".join(failures))
        return 0 if not failures else 1

    report = render_local_stabilization_report(args.summary, args.report)
    summary = json.loads(args.summary.read_text())
    manifest = build_transfer_manifest(
        repo_root,
        canonical_artifacts(repo_root, args.summary, report),
        readiness_statuses={field: summary.get(field) for field in READINESS_FIELDS},
        expected_artifacts=canonical_artifacts(repo_root, args.summary, report),
        regeneration=regeneration_identifiers(repo_root),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{manifest['verification']['status']}: hashed {len(manifest['files'])} artifacts; "
        f"cluster_sweep_authorized={manifest['cluster_sweep_authorized']}"
    )
    return 0 if manifest["verification"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
