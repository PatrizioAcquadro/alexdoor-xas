"""Pure-Python checks for the minimal single-door task USD fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets.door_task import (
    CANONICAL_DOOR_POSES,
    audit_canonical_door_scene_directory,
    canonical_door_scene_path,
    ensure_canonical_door_scenes,
    ensure_diagnostic_door_task_usd,
    ensure_door_task_usd,
    validate_door_task_usd,
)


def test_ensure_door_task_usd_is_deterministic() -> None:
    usd_path = ensure_door_task_usd()
    first_text = usd_path.read_text()

    second_path = ensure_door_task_usd()

    assert usd_path == second_path
    assert usd_path == paths.OUTPUTS_DIR / "door_scene" / "D0.usda"
    assert usd_path.is_file()
    assert second_path.read_text() == first_text


def test_door_task_stage_contract() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    usd_path = ensure_door_task_usd()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)

    assert stage is not None
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    authored_refs = list(_iter_authored_references(stage))
    door_refs = [
        ref
        for ref in authored_refs
        if ref.assetPath and Path(ref.assetPath).expanduser().resolve() == paths.DOOR_USD.resolve()
    ]
    assert len(door_refs) == 1
    assert len(authored_refs) == 1

    assert stage.GetPrimAtPath("/World/DoorTaskDoor/Door").IsValid()
    door_xforms = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "Door" and prim.GetTypeName() == "Xform"
    ]
    assert len(door_xforms) == 1

    hinges = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    assert len(hinges) == 1
    assert hinges[0].GetPath().pathString == "/World/DoorTaskDoor/Doorframe/Hinge"

    frame = stage.GetPrimAtPath("/World/DoorTaskDoor/Doorframe")
    frame_rb = UsdPhysics.RigidBodyAPI(frame)
    assert frame_rb.GetKinematicEnabledAttr().Get() is False

    fixed_joint_prim = stage.GetPrimAtPath("/World/DoorTaskDoor/FixDoorframe")
    assert fixed_joint_prim.IsA(UsdPhysics.FixedJoint)
    fixed_joint = UsdPhysics.FixedJoint(fixed_joint_prim)
    assert fixed_joint.GetBody0Rel().GetTargets() == []
    assert fixed_joint.GetBody1Rel().GetTargets() == [frame.GetPath()]


def test_door_task_stage_excludes_full_scene_references() -> None:
    from pxr import Usd, UsdUtils

    usd_path = ensure_door_task_usd()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    assert stage is not None

    layers, resolved_assets, unresolved_assets = UsdUtils.ComputeAllDependencies(str(usd_path))
    scan_text = "\n".join(
        [
            usd_path.read_text(),
            *(layer.ExportToString() for layer in layers),
            *(str(asset) for asset in resolved_assets),
            *(str(asset) for asset in unresolved_assets),
            *(_reference_text(stage)),
        ]
    ).lower()

    assert "combinedhallwayscene" not in scan_text
    assert "floorplan" not in scan_text
    assert "objects/thor" not in scan_text
    assert "/thor/" not in scan_text
    assert "file:/c:" not in scan_text


def test_door_task_validation_rejects_forbidden_references(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    good_path = ensure_door_task_usd()
    bad_path = tmp_path / "bad_door_task.usda"
    bad_path.write_text(good_path.read_text())

    stage = Usd.Stage.Open(str(bad_path), Usd.Stage.LoadAll)
    assert stage is not None
    bad_prim = UsdGeom.Xform.Define(stage, "/World/ForbiddenSceneRef").GetPrim()
    bad_prim.GetReferences().AddReference(
        "file:/C:/Users/rainb/CombinedHallwayScene/objects/thor/Bad.usda"
    )
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="forbidden scene references"):
        validate_door_task_usd(bad_path)


def _iter_authored_references(stage):
    for prim in stage.TraverseAll():
        refs = prim.GetMetadata("references")
        if refs is None:
            continue
        yield from refs.explicitItems
        yield from refs.addedItems
        yield from refs.prependedItems
        yield from refs.appendedItems


def _reference_text(stage) -> list[str]:
    values: list[str] = []
    for ref in _iter_authored_references(stage):
        values.append(str(ref.assetPath))
        values.append(str(ref.primPath))
    return values


def test_door_task_pose_authoring_pivots_at_doorframe_and_validates() -> None:
    """Posed door-task USD: yaw about the Doorframe pivot + world XY offset.

    The Doorframe (hinge) world position must move by exactly the offset (the
    yaw pivots on it), its orientation must gain the yaw, and the FixDoorframe
    world anchor must follow (validate_door_task_usd re-checks it). The default
    pose must keep authoring a door root without xform ops (byte-stable USD).
    """
    import math

    from pxr import Gf, Usd, UsdGeom

    def frame_world_xf(usd_path):
        stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
        frame = stage.GetPrimAtPath("/World/DoorTaskDoor/Doorframe")
        return stage, UsdGeom.XformCache().GetLocalToWorldTransform(frame)

    default_path = ensure_door_task_usd()
    default_stage, default_xf = frame_world_xf(default_path)
    # The default pose authors no xform opinion of its own on the door root
    # (any ops present are inherited from the referenced Door.usd).
    door_root = default_stage.GetPrimAtPath("/World/DoorTaskDoor")
    stack = door_root.GetAttribute("xformOpOrder").GetPropertyStack(Usd.TimeCode.Default())
    assert all(spec.layer.identifier != str(default_path) for spec in stack)
    pivot = default_xf.ExtractTranslation()

    yaw = 0.10
    offset = (0.02, 0.02)
    posed_path = ensure_door_task_usd("D3")
    assert posed_path == canonical_door_scene_path("D3")
    assert posed_path != default_path

    _, posed_xf = frame_world_xf(posed_path)
    posed_pos = posed_xf.ExtractTranslation()
    expected_pos = pivot + Gf.Vec3d(offset[0], offset[1], 0.0)
    assert Gf.IsClose(posed_pos, expected_pos, 1e-6)

    # Orientation gains exactly the yaw about +Z.
    default_rot = default_xf.RemoveScaleShear().ExtractRotationQuat()
    posed_rot = posed_xf.RemoveScaleShear().ExtractRotationQuat()
    yaw_quat = Gf.Quatd(math.cos(yaw / 2.0), Gf.Vec3d(0.0, 0.0, math.sin(yaw / 2.0)))
    expected_rot = yaw_quat * default_rot
    delta = posed_rot * expected_rot.GetInverse()
    assert abs(abs(delta.GetReal()) - 1.0) < 1e-9

    # The posed file passes the full frozen validation (incl. anchor check).
    validate_door_task_usd(posed_path)


def test_canonical_pose_registry_and_paths_are_exact() -> None:
    assert {
        pose_id: (pose.yaw_rad, pose.xy_offset_m) for pose_id, pose in CANONICAL_DOOR_POSES.items()
    } == {
        "D0": (0.00, (0.00, 0.00)),
        "D1": (+0.05, (+0.02, 0.00)),
        "D2": (-0.05, (0.00, -0.02)),
        "D3": (+0.10, (+0.02, +0.02)),
        "D4": (-0.10, (+0.02, -0.02)),
    }
    assert [canonical_door_scene_path(p).name for p in CANONICAL_DOOR_POSES] == [
        "D0.usda",
        "D1.usda",
        "D2.usda",
        "D3.usda",
        "D4.usda",
    ]


def test_outputs_root_has_only_canonical_top_level_entries() -> None:
    entries = {path.name for path in paths.OUTPUTS_DIR.iterdir()}
    assert {"README.md", "door_scene"} <= entries
    assert entries <= {"README.md", "door_scene", "door_push_alex_v2", "wandb"}
    learned = paths.OUTPUTS_DIR / paths.ALEX_V2_TASK
    if learned.exists():
        assert {path.name for path in learned.iterdir()} <= {"act", "diffusion"}


def test_canonical_scene_directory_audit_rejects_extra_layer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "DOOR_SCENE_DIR", tmp_path / "door_scene")
    ensure_canonical_door_scenes()
    assert sorted(path.name for path in paths.DOOR_SCENE_DIR.iterdir()) == [
        "D0.usda",
        "D1.usda",
        "D2.usda",
        "D3.usda",
        "D4.usda",
    ]
    (paths.DOOR_SCENE_DIR / "D5.usda").write_text("diagnostic")
    with pytest.raises(ValueError, match="must contain exactly"):
        audit_canonical_door_scene_directory()


def test_noncanonical_scene_requires_explicit_cache_path(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(paths, "RUNTIME_CACHE_ROOT", cache_root)
    diagnostic = ensure_diagnostic_door_task_usd(
        cache_root / "door_scenes" / "probe.usda",
        door_yaw_rad=0.123,
        door_xy_offset_m=(0.04, -0.03),
    )
    assert diagnostic.is_file()
    with pytest.raises(ValueError, match="runtime cache"):
        ensure_diagnostic_door_task_usd(
            tmp_path / "outside.usda",
            door_yaw_rad=0.123,
            door_xy_offset_m=(0.04, -0.03),
        )
