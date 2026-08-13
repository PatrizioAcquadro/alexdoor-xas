"""Canonical door-scene generation and validation tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from alexdoor_xas import paths
from alexdoor_xas.assets.door_scene import (
    CANONICAL_DOOR_POSES,
    ensure_door_scene_usd,
    validate_door_scene_usd,
)


def test_door_scene_generation_is_deterministic_and_valid() -> None:
    usd_path = ensure_door_scene_usd()
    first_text = usd_path.read_text()

    assert ensure_door_scene_usd() == usd_path
    assert usd_path == paths.DOOR_SCENE_DIR / "D0.usda"
    assert usd_path.read_text() == first_text

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetPrimAtPath("/World/Door/Door").IsValid()

    hinges = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert [prim.GetPath().pathString for prim in hinges] == [
        "/World/Door/Doorframe/Hinge"
    ]
    validate_door_scene_usd(usd_path)


def test_validation_rejects_forbidden_references(tmp_path: Path) -> None:
    good_path = ensure_door_scene_usd()
    bad_path = tmp_path / "bad_door_scene.usda"
    bad_path.write_text(good_path.read_text())

    stage = Usd.Stage.Open(str(bad_path), Usd.Stage.LoadAll)
    bad_prim = UsdGeom.Xform.Define(stage, "/World/ForbiddenSceneRef").GetPrim()
    bad_prim.GetReferences().AddReference(
        "file:/C:/Users/rainb/CombinedHallwayScene/objects/thor/Bad.usda"
    )
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="forbidden scene references"):
        validate_door_scene_usd(bad_path)


def test_posed_scene_rotates_about_the_hinge_and_moves_the_anchor() -> None:
    def frame_world_xf(usd_path):
        stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
        frame = stage.GetPrimAtPath("/World/Door/Doorframe")
        return stage, UsdGeom.XformCache().GetLocalToWorldTransform(frame)

    default_path = ensure_door_scene_usd()
    default_stage, default_xf = frame_world_xf(default_path)
    door_root = default_stage.GetPrimAtPath("/World/Door")
    stack = door_root.GetAttribute("xformOpOrder").GetPropertyStack(Usd.TimeCode.Default())
    assert all(spec.layer.identifier != str(default_path) for spec in stack)

    posed_path = ensure_door_scene_usd("D3")
    _, posed_xf = frame_world_xf(posed_path)
    expected_pos = default_xf.ExtractTranslation() + Gf.Vec3d(0.02, 0.02, 0.0)
    assert Gf.IsClose(posed_xf.ExtractTranslation(), expected_pos, 1e-6)

    default_rot = default_xf.RemoveScaleShear().ExtractRotationQuat()
    posed_rot = posed_xf.RemoveScaleShear().ExtractRotationQuat()
    yaw_quat = Gf.Quatd(math.cos(0.05), Gf.Vec3d(0.0, 0.0, math.sin(0.05)))
    delta = posed_rot * (yaw_quat * default_rot).GetInverse()
    assert abs(abs(delta.GetReal()) - 1.0) < 1e-9
    validate_door_scene_usd(posed_path)


def test_canonical_pose_registry_is_exact() -> None:
    assert {
        pose_id: (pose.yaw_rad, pose.xy_offset_m) for pose_id, pose in CANONICAL_DOOR_POSES.items()
    } == {
        "D0": (0.00, (0.00, 0.00)),
        "D1": (+0.05, (+0.02, 0.00)),
        "D2": (-0.05, (0.00, -0.02)),
        "D3": (+0.10, (+0.02, +0.02)),
        "D4": (-0.10, (+0.02, -0.02)),
    }
