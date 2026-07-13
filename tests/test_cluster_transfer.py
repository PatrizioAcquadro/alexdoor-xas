from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from scripts.build_cluster_transfer_manifest import (
    REQUIRED_CATEGORIES,
    SCHEMA,
    build_transfer_manifest,
    verify_transfer_manifest,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Transfer Test")
    _git(tmp_path, "config", "user.email", "transfer@example.invalid")
    artifacts: dict[str, list[Path]] = {}
    for category in REQUIRED_CATEGORIES:
        count = 2 if category == "evaluation" else 1
        artifacts[category] = []
        for index in range(count):
            path = tmp_path / category / f"artifact_{index}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{category}:{index}".encode())
            artifacts[category].append(path)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "fixture")
    statuses = {
        "metadata_coverage": "PASS",
        "protocol_consistency": "PASS",
        "safety_readiness": "PASS",
    }
    manifest = build_transfer_manifest(
        tmp_path,
        artifacts,
        readiness_statuses=statuses,
        expected_artifacts=artifacts,
        regeneration={"fixture": "cluster-transfer-v1"},
    )
    return artifacts, manifest


def _failures(manifest, root: Path, artifacts) -> list[str]:
    return verify_transfer_manifest(manifest, root, expected_artifacts=artifacts)


def test_transfer_manifest_round_trip_hashes_complete_inventory(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)

    assert manifest["schema"] == SCHEMA
    assert manifest["verification"]["status"] == "PASS"
    assert manifest["cluster_sweep_authorized"] is True
    assert manifest["source_git"]["clean_tree"] is True
    assert manifest["source_git"]["commit"] == _git(tmp_path, "rev-parse", "HEAD")
    assert manifest["category_counts"]["evaluation"] == 2
    assert _failures(json.loads(json.dumps(manifest)), tmp_path, artifacts) == []


@pytest.mark.parametrize("schema", [None, "wrong.schema"])
def test_verifier_rejects_missing_or_wrong_schema(tmp_path, schema) -> None:
    artifacts, manifest = _fixture(tmp_path)
    if schema is None:
        manifest.pop("schema")
    else:
        manifest["schema"] = schema
    assert any("schema" in failure for failure in _failures(manifest, tmp_path, artifacts))


@pytest.mark.parametrize("files", [None, []])
def test_verifier_rejects_missing_or_empty_files(tmp_path, files) -> None:
    artifacts, manifest = _fixture(tmp_path)
    if files is None:
        manifest.pop("files")
    else:
        manifest["files"] = files
    failures = _failures(manifest, tmp_path, artifacts)
    assert any("files array" in failure for failure in failures)
    assert manifest.get("cluster_sweep_authorized") is True  # forged stale value is not trusted


def test_verifier_rejects_missing_and_unexpected_categories(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["category"] != "report"
    ]
    manifest["files"][0]["category"] = "mystery"
    failures = _failures(manifest, tmp_path, artifacts)
    assert any("missing required categories" in failure for failure in failures)
    assert any("unexpected categories" in failure for failure in failures)


def test_verifier_rejects_inconsistent_category_counts(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    manifest["category_counts"]["evaluation"] = 999
    assert any(
        "category_counts" in failure for failure in _failures(manifest, tmp_path, artifacts)
    )


def test_verifier_rejects_duplicate_path_same_or_multiple_categories(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    duplicate = copy.deepcopy(manifest["files"][0])
    manifest["files"].append(duplicate)
    assert any("duplicate path" in f for f in _failures(manifest, tmp_path, artifacts))

    artifacts, manifest = _fixture(tmp_path / "multi")
    duplicate = copy.deepcopy(manifest["files"][0])
    duplicate["category"] = "report"
    manifest["files"].append(duplicate)
    failures = _failures(manifest, tmp_path / "multi", artifacts)
    assert any("multiple categories" in f for f in failures)


def test_verifier_rejects_escaping_and_missing_paths(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    manifest["files"][0]["path"] = "../escape.bin"
    assert any("escapes" in f for f in _failures(manifest, tmp_path, artifacts))

    artifacts, manifest = _fixture(tmp_path / "missing")
    Path(artifacts["dataset"][0]).unlink()
    assert any("missing file" in f for f in _failures(manifest, tmp_path / "missing", artifacts))


def test_verifier_rejects_size_hash_and_malformed_sha256(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    manifest["files"][0]["size_bytes"] += 1
    assert any("size mismatch" in f for f in _failures(manifest, tmp_path, artifacts))

    artifacts, manifest = _fixture(tmp_path / "hash")
    artifacts["dataset"][0].write_bytes(b"same-size-ish")
    failures = _failures(manifest, tmp_path / "hash", artifacts)
    assert any("sha256 mismatch" in f or "size mismatch" in f for f in failures)

    artifacts, manifest = _fixture(tmp_path / "malformed")
    manifest["files"][0]["sha256"] = "not-a-sha"
    failures = _failures(manifest, tmp_path / "malformed", artifacts)
    assert any("malformed sha256" in f for f in failures)


def test_verifier_rejects_incomplete_canonical_inventory(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    removed = manifest["files"].pop()
    assert removed["category"] in REQUIRED_CATEGORIES
    failures = _failures(manifest, tmp_path, artifacts)
    assert any("canonical artifact inventory" in f for f in failures)


@pytest.mark.parametrize("source_git", [None, {"commit": "0" * 40}])
def test_verifier_rejects_missing_or_mismatched_source_commit(tmp_path, source_git) -> None:
    artifacts, manifest = _fixture(tmp_path)
    if source_git is None:
        manifest.pop("source_git")
    else:
        manifest["source_git"] = source_git
    failures = _failures(manifest, tmp_path, artifacts)
    assert any("source Git" in f or "source commit" in f for f in failures)


def test_dirty_tree_cannot_generate_authorization_bearing_manifest(tmp_path) -> None:
    artifacts, manifest = _fixture(tmp_path)
    artifacts["dataset"][0].write_bytes(b"dirty")
    dirty = build_transfer_manifest(
        tmp_path,
        artifacts,
        readiness_statuses={
            "metadata_coverage": "PASS",
            "protocol_consistency": "PASS",
            "safety_readiness": "PASS",
        },
        expected_artifacts=artifacts,
        regeneration={"fixture": "cluster-transfer-v1"},
    )
    assert dirty["source_git"]["clean_tree"] is False
    assert dirty["cluster_sweep_authorized"] is False
    assert dirty["verification"]["status"] == "FAIL"
    assert any("source tree is dirty" in f for f in dirty["verification"]["failures"])
    assert manifest["cluster_sweep_authorized"] is True


def test_authorization_is_derived_from_all_readiness_gates(tmp_path) -> None:
    artifacts, _ = _fixture(tmp_path)
    blocked = build_transfer_manifest(
        tmp_path,
        artifacts,
        readiness_statuses={
            "metadata_coverage": "PASS",
            "protocol_consistency": "PASS",
            "safety_readiness": "REVIEW_REQUIRED",
        },
        expected_artifacts=artifacts,
        regeneration={"fixture": "cluster-transfer-v1"},
    )
    assert blocked["verification"]["status"] == "PASS"
    assert blocked["cluster_sweep_authorized"] is False
