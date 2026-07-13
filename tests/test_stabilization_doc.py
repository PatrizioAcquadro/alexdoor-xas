"""Artifact-consistency checks for stabilization evidence in the project status."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_stabilization_doc.py"
    spec = importlib.util.spec_from_file_location("verify_stabilization_doc_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(root: Path, relative: Path, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _fixture(root: Path) -> tuple[object, dict[str, str]]:
    module = _module()
    policy = {
        "version": "alexdoor.warning-adjudication.v3",
        "a2.joint_velocity_limit": {
            "evidence_basis": {
                "artifact_warning_events": 876,
                "artifact_warning_events_per_cell": 219,
                "max_observed_count_per_joint_per_rollout": 4,
                "max_observed_warning_records_per_rollout": 11,
            }
        },
    }
    summary = {
        "runs": {
            "local_smoke_act_a2_n50_seed0": {
                "safety_readiness": {"warning_adjudication_policy": policy}
            }
        }
    }
    _write_json(root, module.SUMMARY, summary)
    for relative, fingerprint in ((module.A2_EVAL, "a2hash"), (module.A3_EVAL, "a3hash")):
        _write_json(
            root,
            relative,
            {
                "dataset_provenance": {
                    "checkpoint_dataset_fingerprint_sha256": fingerprint
                }
            },
        )
    _write_json(root, module.A2_TRAIN, {"history": {"best_val_l1": 0.030836489}})
    _write_json(root, module.A3_TRAIN, {"history": {"best_val_l1": 0.032292170}})
    return module, module.required_document_tokens(root)


def test_stabilization_document_matches_artifact_tokens(tmp_path: Path) -> None:
    module, tokens = _fixture(tmp_path)
    document = tmp_path / module.DOCUMENT
    document.parent.mkdir(parents=True)
    document.write_text("\n".join(tokens.values()))
    assert module.verify_document(tmp_path) == []


def test_stabilization_document_reports_stale_artifact_value(tmp_path: Path) -> None:
    module, tokens = _fixture(tmp_path)
    document = tmp_path / module.DOCUMENT
    document.parent.mkdir(parents=True)
    document.write_text(
        "\n".join(
            value
            for label, value in tokens.items()
            if label != "A3 exact dataset fingerprint"
        )
    )
    failures = module.verify_document(tmp_path)
    assert failures == ["A3 exact dataset fingerprint is stale or missing (expected 'a3hash')"]
