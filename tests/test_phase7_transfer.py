from __future__ import annotations

import json

from scripts.build_phase7_transfer_manifest import (
    build_transfer_manifest,
    verify_transfer_manifest,
)


def test_transfer_manifest_hashes_and_verifies_every_category(tmp_path) -> None:
    artifacts = {}
    for category in (
        "dataset",
        "split",
        "norm_stats",
        "checkpoint",
        "evaluation",
        "summary",
        "report",
    ):
        path = tmp_path / category / "artifact.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(category.encode())
        artifacts[category] = [path]

    manifest = build_transfer_manifest(tmp_path, artifacts, phase7_authorized=False)

    assert manifest["verification"]["status"] == "PASS"
    assert manifest["phase7_authorized"] is False
    assert {entry["category"] for entry in manifest["files"]} == set(artifacts)
    assert verify_transfer_manifest(manifest, tmp_path) == []

    (tmp_path / "dataset" / "artifact.bin").write_bytes(b"tampered")
    failures = verify_transfer_manifest(manifest, tmp_path)
    assert any("sha256 mismatch" in failure for failure in failures)


def test_transfer_manifest_rejects_missing_required_category(tmp_path) -> None:
    artifacts = {"dataset": []}
    try:
        build_transfer_manifest(tmp_path, artifacts, phase7_authorized=False)
    except ValueError as error:
        assert "required artifact categories" in str(error)
    else:  # pragma: no cover - assertion form keeps the dependency surface tiny
        raise AssertionError("missing categories were accepted")


def test_written_manifest_is_json_round_trip_safe(tmp_path) -> None:
    artifacts = {}
    for category in (
        "dataset",
        "split",
        "norm_stats",
        "checkpoint",
        "evaluation",
        "summary",
        "report",
    ):
        path = tmp_path / f"{category}.json"
        path.write_text(json.dumps({"category": category}))
        artifacts[category] = [path]
    manifest = build_transfer_manifest(tmp_path, artifacts, phase7_authorized=True)
    assert json.loads(json.dumps(manifest)) == manifest
