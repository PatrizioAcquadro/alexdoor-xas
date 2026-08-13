"""Canonical and diagnostic single-door USD layers for AlexDoor-XAS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from alexdoor_xas import paths


@dataclass(frozen=True)
class _DoorPose:
    pose_id: str
    yaw_rad: float
    xy_offset_m: tuple[float, float]


_CANONICAL_DOOR_POSES = {
    "D0": _DoorPose("D0", 0.00, (0.00, 0.00)),
    "D1": _DoorPose("D1", +0.05, (+0.02, 0.00)),
    "D2": _DoorPose("D2", -0.05, (0.00, -0.02)),
    "D3": _DoorPose("D3", +0.10, (+0.02, +0.02)),
    "D4": _DoorPose("D4", -0.10, (+0.02, -0.02)),
}
CANONICAL_DOOR_POSES: Mapping[str, _DoorPose] = MappingProxyType(_CANONICAL_DOOR_POSES)
DEFAULT_DOOR_POSE_ID = "D0"

# Pure PXR cannot resolve Isaac's built-in OmniPBR material.
_ALLOWED_UNRESOLVED_ASSETS = {"OmniPBR.mdl"}
_FORBIDDEN_REFERENCES = (
    "file:/c:",
    "combinedhallwayscene",
    "floorplan",
    "objects/thor",
    "/thor/",
    "\\thor\\",
)


def canonical_door_pose(pose_id: str = DEFAULT_DOOR_POSE_ID) -> _DoorPose:
    """Return a frozen pose by ID."""
    try:
        return CANONICAL_DOOR_POSES[pose_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown canonical door pose {pose_id!r}; expected one of "
            f"{tuple(CANONICAL_DOOR_POSES)}"
        ) from exc


def ensure_door_scene_usd(pose_id: str = DEFAULT_DOOR_POSE_ID) -> Path:
    """Create or refresh one canonical D0-D4 scene layer."""
    pose = canonical_door_pose(pose_id)
    path = paths.DOOR_SCENE_DIR / f"{pose_id}.usda"
    return _ensure_door_scene_usd(path, pose.yaw_rad, pose.xy_offset_m)


def _ensure_door_scene_usd(
    usd_path: Path,
    door_yaw_rad: float,
    door_xy_offset_m: tuple[float, float],
) -> Path:
    """Author one deterministic layer and replace it only when content changes."""
    usd_path = usd_path.expanduser()
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = usd_path.with_name(f".{usd_path.stem}.tmp.usda")
    if tmp_path.exists():
        tmp_path.unlink()

    _author_door_scene_usd(tmp_path, door_yaw_rad, tuple(door_xy_offset_m))
    new_text = tmp_path.read_text()
    old_text = usd_path.read_text() if usd_path.exists() else None
    if old_text != new_text:
        tmp_path.replace(usd_path)
    else:
        tmp_path.unlink()

    validate_door_scene_usd(usd_path)
    return usd_path


def validate_door_scene_usd(path: str | Path) -> None:
    """Validate a canonical door-scene USD without launching Isaac Kit."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    usd_path = Path(path).expanduser()
    if not usd_path.is_file():
        raise FileNotFoundError(f"door scene USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"could not open door scene USD: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim or default_prim.GetPath().pathString != "/World":
        raise ValueError(f"door scene USD must use /World as defaultPrim: {usd_path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"door scene USD must be Z-up: {usd_path}")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError(f"door scene USD must use metersPerUnit=1.0: {usd_path}")

    authored_refs = list(_iter_authored_references(stage))
    door_refs = [
        ref
        for ref in authored_refs
        if ref.assetPath and Path(ref.assetPath).expanduser().resolve() == paths.DOOR_USD.resolve()
    ]
    if len(door_refs) != 1:
        raise ValueError(
            f"door scene USD must author exactly one reference to {paths.DOOR_USD}: {usd_path}"
        )

    if not stage.GetPrimAtPath("/World/Door").IsValid():
        raise ValueError("door scene USD is missing /World/Door")
    if not stage.GetPrimAtPath("/World/Door/Door").IsValid():
        raise ValueError("door scene USD is missing /World/Door/Door")

    door_xforms = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "Door"
        and prim.GetPath().pathString != "/World/Door"
        and prim.GetTypeName() == "Xform"
    ]
    if len(door_xforms) != 1:
        raise ValueError(f"door scene USD must contain exactly one door object: {usd_path}")

    hinges = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    if len(hinges) != 1:
        raise ValueError(f"door scene USD must contain exactly one revolute hinge: {usd_path}")

    _validate_physics_overrides(stage, usd_path)
    _validate_dependencies(usd_path, stage, UsdUtils)


def _author_door_scene_usd(
    path: Path,
    door_yaw_rad: float = 0.0,
    door_xy_offset_m: tuple[float, float] = (0.0, 0.0),
) -> None:
    import math

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    if not paths.DOOR_USD.is_file():
        raise FileNotFoundError(f"source door USD not found: {paths.DOOR_USD}")

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor_xform = UsdGeom.Xformable(floor.GetPrim())
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    floor_xform.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 0.05))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(400.0)

    door_root = UsdGeom.Xform.Define(stage, "/World/Door").GetPrim()
    door_root.GetReferences().AddReference(str(paths.DOOR_USD), Sdf.Path("/DoorObject"))
    UsdPhysics.ArticulationRootAPI.Apply(door_root)

    if door_yaw_rad != 0.0 or door_xy_offset_m != (0.0, 0.0):
        # Rotate around the hinge, then translate in world XY. D0 stays byte-stable.
        unposed_frame = stage.GetPrimAtPath("/World/Door/Doorframe")
        pivot = Gf.Vec3d(
            UsdGeom.XformCache().GetLocalToWorldTransform(unposed_frame).ExtractTranslation()
        )
        offset = Gf.Vec3d(door_xy_offset_m[0], door_xy_offset_m[1], 0.0)
        pose = (
            Gf.Matrix4d().SetTranslate(-pivot)
            * Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(door_yaw_rad))
            )
            * Gf.Matrix4d().SetTranslate(pivot + offset)
        )
        door_xf = UsdGeom.Xformable(door_root)
        existing_local = door_xf.GetLocalTransformation()
        door_xf.MakeMatrixXform().Set(existing_local * pose)

    frame = stage.OverridePrim("/World/Door/Doorframe")
    frame_rb = UsdPhysics.RigidBodyAPI.Apply(frame)
    frame_rb.CreateRigidBodyEnabledAttr(True)
    frame_rb.CreateKinematicEnabledAttr(False)
    frame_rb.CreateStartsAsleepAttr(True)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Door/FixDoorframe")
    fixed_joint.CreateJointEnabledAttr(True)
    fixed_joint.CreateBody1Rel().SetTargets([frame.GetPath()])
    # An explicit world anchor prevents namespaced scenes from snapping to the origin.
    frame_xf = UsdGeom.XformCache().GetLocalToWorldTransform(frame)
    frame_rot = frame_xf.RemoveScaleShear().ExtractRotationQuat()
    fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(frame_xf.ExtractTranslation()))
    fixed_joint.CreateLocalRot0Attr(Gf.Quatf(frame_rot))
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    _apply_mass_override(stage.OverridePrim("/World/Door/Door"), 25.0, (1.0, 1.0, 1.0))
    _apply_mass_override(stage.OverridePrim("/World/Door/Handle"), 1.0, (0.01, 0.01, 0.01))

    stage.GetRootLayer().Save()


def _apply_mass_override(prim, mass: float, diagonal_inertia: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdPhysics

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _validate_physics_overrides(stage, usd_path: Path) -> None:
    from pxr import UsdPhysics

    frame = stage.GetPrimAtPath("/World/Door/Doorframe")
    frame_rb = UsdPhysics.RigidBodyAPI(frame)
    if frame_rb.GetKinematicEnabledAttr().Get() is not False:
        raise ValueError(f"door scene layer must keep the door frame non-kinematic: {usd_path}")

    fixed_joint_prim = stage.GetPrimAtPath("/World/Door/FixDoorframe")
    if not fixed_joint_prim.IsA(UsdPhysics.FixedJoint):
        raise ValueError(f"door scene layer must fix the door frame to world: {usd_path}")
    fixed_joint = UsdPhysics.FixedJoint(fixed_joint_prim)
    if fixed_joint.GetBody0Rel().GetTargets():
        raise ValueError(f"door frame fixed joint must use the world as body0: {usd_path}")
    if list(fixed_joint.GetBody1Rel().GetTargets()) != [frame.GetPath()]:
        raise ValueError(
            f"door frame fixed joint must target /World/Door/Doorframe: {usd_path}"
        )
    _validate_world_joint_anchor(fixed_joint, frame, usd_path)

    for prim_path in ("/World/Door/Door", "/World/Door/Handle"):
        prim = stage.GetPrimAtPath(prim_path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass is None or mass <= 0:
            raise ValueError(f"{prim_path} must have a positive scene-layer mass: {usd_path}")
        if inertia is None or any(value <= 0 for value in inertia):
            raise ValueError(f"{prim_path} must have positive scene-layer inertia: {usd_path}")


def _validate_world_joint_anchor(fixed_joint, frame, usd_path: Path) -> None:
    """The world-side joint anchor must sit at the door frame's composed pose."""
    from pxr import Gf, UsdGeom

    local_pos0 = fixed_joint.GetLocalPos0Attr().Get()
    local_rot0 = fixed_joint.GetLocalRot0Attr().Get()
    if local_pos0 is None or local_rot0 is None:
        raise ValueError(
            f"door frame fixed joint must author explicit world-side frames: {usd_path}"
        )
    frame_xf = UsdGeom.XformCache().GetLocalToWorldTransform(frame)
    expected_pos = Gf.Vec3f(frame_xf.ExtractTranslation())
    if not Gf.IsClose(Gf.Vec3f(local_pos0), expected_pos, 1e-5):
        raise ValueError(
            "door frame fixed joint world anchor must match the frame's world pose "
            f"({expected_pos}), got {local_pos0}: {usd_path}"
        )
    # Matching only position would twist a yawed door at solve time.
    expected_rot = Gf.Quatf(frame_xf.RemoveScaleShear().ExtractRotationQuat())
    delta = Gf.Quatf(local_rot0) * expected_rot.GetInverse()
    if abs(abs(delta.GetReal()) - 1.0) > 1e-5:
        raise ValueError(
            "door frame fixed joint world anchor rotation must match the frame's "
            f"world orientation ({expected_rot}), got {local_rot0}: {usd_path}"
        )


def _validate_dependencies(usd_path: Path, stage, usd_utils) -> None:
    layers, resolved_assets, unresolved_assets = usd_utils.ComputeAllDependencies(str(usd_path))
    scan_text = "\n".join(
        [
            str(usd_path),
            str(paths.DOOR_USD),
            *(str(asset) for asset in resolved_assets),
            *(str(asset) for asset in unresolved_assets),
            *(_layer_text(layer) for layer in layers),
            *(_reference_text(stage)),
        ]
    ).lower()
    offenders = [token for token in _FORBIDDEN_REFERENCES if token in scan_text]
    if offenders:
        raise ValueError(f"door scene USD contains forbidden scene references: {offenders}")

    bad_unresolved = [
        asset
        for asset in unresolved_assets
        if Path(str(asset)).name not in _ALLOWED_UNRESOLVED_ASSETS
    ]
    if bad_unresolved:
        raise ValueError(f"door scene USD has unresolved references: {bad_unresolved}")


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


def _layer_text(layer) -> str:
    try:
        return layer.ExportToString()
    except Exception:  # noqa: BLE001 - validation must surface forbidden refs if possible.
        return str(layer.identifier)


__all__ = [
    "CANONICAL_DOOR_POSES",
    "DEFAULT_DOOR_POSE_ID",
    "canonical_door_pose",
    "ensure_door_scene_usd",
    "validate_door_scene_usd",
]
