"""Pure tests for the static Alex V2 URDF manifest builder."""

from __future__ import annotations

import hashlib

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import EXPECTED_RUNTIME_JOINTS
from alexdoor_xas.assets.alex_v2_manifest import (
    EXPECTED_ALEX_V2_URDF_SHA256,
    EXPECTED_COLLISION_RECORD_COUNT,
    AlexV2ManifestError,
    build_alex_v2_manifest,
)


def test_builder_derives_identity_joints_and_primitive_collisions() -> None:
    manifest = build_alex_v2_manifest()

    assert hashlib.sha256(paths.ALEX_V2_URDF.read_bytes()).hexdigest() == (
        EXPECTED_ALEX_V2_URDF_SHA256
    )
    assert manifest["urdf_sha256"] == EXPECTED_ALEX_V2_URDF_SHA256
    assert manifest["robot_asset_sha256"] == EXPECTED_ALEX_V2_URDF_SHA256
    assert manifest["movable_joint_count"] == 29
    assert len(manifest["movable_joints"]) == len(set(manifest["movable_joints"]))
    assert set(manifest["movable_joints"]) == set(EXPECTED_RUNTIME_JOINTS)

    links = manifest["collision_profile"]["links"]
    assert len(links) == 19
    assert sum(len(records) for records in links.values()) == EXPECTED_COLLISION_RECORD_COUNT
    right_gripper = links["RIGHT_GRIPPER_Z_LINK"]
    assert [record["name"] for record in right_gripper] == [
        "right_gripper_z_collision",
        "right_fist_collision",
        "right_finger_collision",
        "right_thumb_collision",
    ]
    assert {record["shape"] for records in links.values() for record in records} == {
        "box",
        "capsule",
        "cylinder",
        "sphere",
    }


def test_builder_rejects_any_urdf_identity_drift(tmp_path) -> None:
    changed = tmp_path / "alex_v2.urdf"
    changed.write_bytes(paths.ALEX_V2_URDF.read_bytes() + b"\n")

    with pytest.raises(AlexV2ManifestError, match="identity differs"):
        build_alex_v2_manifest(changed)
