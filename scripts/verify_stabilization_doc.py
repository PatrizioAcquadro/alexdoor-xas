#!/usr/bin/env python
"""Verify the stabilization evidence in the project status against final artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from alexdoor_xas import paths

SUMMARY = Path("outputs/local_smoke_n50/summary.json")
A2_EVAL = Path(
    "outputs/door_push_alex_v2/act_door_push/"
    "local_smoke_act_a2_n50_seed0/metrics/act_eval_D0.json"
)
A3_EVAL = Path(
    "outputs/door_push_alex_v2/act_door_push/"
    "local_smoke_act_a3_n50_seed0/metrics/act_eval_D0.json"
)
A2_TRAIN = Path(
    "outputs/door_push_alex_v2/act_door_push/"
    "local_smoke_act_a2_n50_seed0/logs/train_log.json"
)
A3_TRAIN = Path(
    "outputs/door_push_alex_v2/act_door_push/"
    "local_smoke_act_a3_n50_seed0/logs/train_log.json"
)
DOCUMENT = Path("knowledge/wiki/status.md")


def _load_json(root: Path, relative: Path) -> dict:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return json.loads(path.read_text())


def _best_val_l1(train_log: dict) -> float:
    history = train_log.get("history")
    if isinstance(history, dict) and history.get("best_val_l1") is not None:
        return float(history["best_val_l1"])
    values = [
        float(row["val_l1"])
        for row in history or []
        if row.get("val_l1") is not None
    ]
    if not values:
        raise ValueError("ACT training log has no validation L1 values")
    return min(values)


def required_document_tokens(root: Path) -> dict[str, str]:
    summary = _load_json(root, SUMMARY)
    a2_eval = _load_json(root, A2_EVAL)
    a3_eval = _load_json(root, A3_EVAL)
    a2_train = _load_json(root, A2_TRAIN)
    a3_train = _load_json(root, A3_TRAIN)
    policy = summary["runs"]["local_smoke_act_a2_n50_seed0"]["safety_readiness"][
        "warning_adjudication_policy"
    ]
    basis = policy["a2.joint_velocity_limit"]["evidence_basis"]
    return {
        "A2 exact dataset fingerprint": a2_eval["dataset_provenance"][
            "checkpoint_dataset_fingerprint_sha256"
        ],
        "A3 exact dataset fingerprint": a3_eval["dataset_provenance"][
            "checkpoint_dataset_fingerprint_sha256"
        ],
        "ACT-A2 best validation L1": f"{_best_val_l1(a2_train):.8f}",
        "ACT-A3 best validation L1": f"{_best_val_l1(a3_train):.8f}",
        "warning policy version": policy["version"].rsplit(".", maxsplit=1)[-1],
        "total per-joint warnings": f"All {basis['artifact_warning_events']} warnings",
        "warnings per matrix cell": f"{basis['artifact_warning_events_per_cell']} in each",
        "max per-joint count": (
            f"≤{basis['max_observed_count_per_joint_per_rollout']} events for one joint"
        ),
        "max total count": (
            f"≤{basis['max_observed_warning_records_per_rollout']} total"
        ),
    }


def verify_document(root: Path) -> list[str]:
    document_path = root / DOCUMENT
    if not document_path.is_file():
        return [f"required document is missing: {document_path}"]
    document = document_path.read_text()
    failures = []
    for label, token in required_document_tokens(root).items():
        if token not in document:
            failures.append(f"{label} is stale or missing (expected {token!r})")
    return failures


def main() -> int:
    failures = verify_document(paths.REPO_ROOT)
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: project status stabilization evidence matches final artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
