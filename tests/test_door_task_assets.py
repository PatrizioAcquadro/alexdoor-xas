"""Pure-Python checks for the minimal single-door task USD fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets.door_task import (
    ensure_door_task_usd,
    validate_door_task_usd,
)


def test_ensure_door_task_usd_is_deterministic() -> None:
    usd_path = ensure_door_task_usd()
    first_text = usd_path.read_text()

    second_path = ensure_door_task_usd()

    assert usd_path == second_path
    assert usd_path == paths.OUTPUTS_DIR / "door_task" / "door_task.usda"
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
