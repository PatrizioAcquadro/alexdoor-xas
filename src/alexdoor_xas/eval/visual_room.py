"""Presentation-only room context for learned-policy rollout videos.

The physics task remains the calibrated single-door D0 environment.  This
module references one room and the hallway from the furnished combined scene,
aligns their doorway around the existing task door, removes the duplicate
room door, and disables every imported physics schema.  The result changes
pixels only: observations, control, contact filtering, and success semantics
stay bound to the original ``DoorTaskDoor``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths


class VisualRoomError(RuntimeError):
    """A requested presentation room could not be composed safely."""


@dataclass(frozen=True)
class VisualRoomProfile:
    """Frozen placement and camera contract for one combined-scene room."""

    name: str
    label: str
    room_source_prim: str
    hallway_source_prim: str
    source_door_subpath: str
    context_yaw_deg: float
    camera_eye: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    ambient_light_intensity: float


_PROFILES = {
    "floorplan212_living_room": VisualRoomProfile(
        name="floorplan212_living_room",
        label="FloorPlan212 living room",
        room_source_prim="/World/scene_01",
        hallway_source_prim="/World/Hallway",
        source_door_subpath="Door",
        # FloorPlan212 lies on the negative-X side of its hallway doorway.
        # Rotating the visual context by 180 degrees places that interior on
        # Alex's positive-X side while the corridor remains beyond the door.
        context_yaw_deg=180.0,
        camera_eye=(1.85, 1.15, 2.05),
        camera_lookat=(-0.70, -0.05, 0.92),
        ambient_light_intensity=450.0,
    ),
}

VISUAL_ROOM_PROFILE_NAMES = tuple(_PROFILES)
DEFAULT_VISUAL_ROOM_PROFILE = VISUAL_ROOM_PROFILE_NAMES[0]
VISUAL_ROOM_RENDER_WARMUP_FRAMES = 8


def visual_room_profile(name: str) -> VisualRoomProfile:
    """Return one known room profile, rejecting silent fallback."""
    try:
        return _PROFILES[name]
    except KeyError as error:
        raise VisualRoomError(
            f"unknown visual room {name!r}; expected one of {list(VISUAL_ROOM_PROFILE_NAMES)}"
        ) from error


def attach_visual_room(
    stage: Any,
    profile_name: str,
    *,
    target_doorframe_path: str,
    root_path: str = "/World/VisualRoom",
) -> dict[str, Any]:
    """Compose and align one room+hallway context with all imported physics off."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: PLC0415

    profile = visual_room_profile(profile_name)
    scene_path = paths.COMBINED_SCENE_USD.expanduser().resolve()
    if not scene_path.is_file():
        raise VisualRoomError(f"combined scene is unavailable: {scene_path}")
    if stage is None:
        raise VisualRoomError("active USD stage is unavailable")
    if stage.GetPrimAtPath(root_path).IsValid():
        raise VisualRoomError(f"visual room root already exists: {root_path}")

    target_frame = stage.GetPrimAtPath(target_doorframe_path)
    if not target_frame.IsValid():
        raise VisualRoomError(f"task doorframe is unavailable: {target_doorframe_path}")

    context = UsdGeom.Xform.Define(stage, root_path)
    room = UsdGeom.Xform.Define(stage, f"{root_path}/Room")
    hallway = UsdGeom.Xform.Define(stage, f"{root_path}/Hallway")
    room.GetPrim().GetReferences().AddReference(
        str(scene_path), Sdf.Path(profile.room_source_prim)
    )
    hallway.GetPrim().GetReferences().AddReference(
        str(scene_path), Sdf.Path(profile.hallway_source_prim)
    )
    fill_light = UsdLux.DomeLight.Define(stage, f"{root_path}/VideoFillLight")
    fill_light.CreateIntensityAttr(profile.ambient_light_intensity)
    fill_light.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.84))

    source_door_path = f"{root_path}/Room/{profile.source_door_subpath}"
    source_frame_path = f"{source_door_path}/Doorframe"
    source_frame = stage.GetPrimAtPath(source_frame_path)
    if not source_frame.IsValid():
        raise VisualRoomError(
            f"selected room doorframe did not compose at {source_frame_path}"
        )

    cache = UsdGeom.XformCache()
    source_world = cache.GetLocalToWorldTransform(source_frame)
    target_world = cache.GetLocalToWorldTransform(target_frame).RemoveScaleShear()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    source_center_world = (
        bounds.ComputeWorldBound(source_frame).ComputeAlignedRange().GetMidpoint()
    )
    target_center_world = (
        bounds.ComputeWorldBound(target_frame).ComputeAlignedRange().GetMidpoint()
    )
    source_center_local = source_world.GetInverse().Transform(source_center_world)
    room_side_rotation = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), profile.context_yaw_deg)
    )
    desired_source_world = room_side_rotation * target_world
    predicted_center_world = desired_source_world.Transform(source_center_local)
    center_correction = target_center_world - predicted_center_world
    desired_source_world.SetTranslateOnly(
        desired_source_world.ExtractTranslation() + center_correction
    )
    alignment = source_world.GetInverse() * desired_source_world
    context.MakeMatrixXform().Set(alignment)

    alignment_report = _validate_alignment(
        stage,
        source_frame_path=source_frame_path,
        target_frame_path=target_doorframe_path,
        expected_yaw_deg=profile.context_yaw_deg,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )

    # The calibrated task door occupies this opening.  Removing the room copy
    # prevents duplicate panels/frames while preserving the surrounding wall.
    stage.OverridePrim(source_door_path).SetActive(False)

    physics_counts = _disable_imported_physics(
        stage, root_path, UsdPhysics=UsdPhysics
    )

    return {
        "mode": "presentation_only_visual_context",
        "physics_enabled": False,
        "profile": asdict(profile),
        "source_asset": str(scene_path),
        "source_asset_sha256": _sha256(scene_path),
        "root_path": root_path,
        "selected_room_path": f"{root_path}/Room",
        "hallway_path": f"{root_path}/Hallway",
        "removed_duplicate_door_path": source_door_path,
        "target_task_doorframe_path": target_doorframe_path,
        "doorway_alignment": alignment_report,
        "disabled_physics": physics_counts,
    }


def _disable_imported_physics(stage, root_path: str, *, UsdPhysics) -> dict[str, int]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise VisualRoomError(f"visual room root is unavailable: {root_path}")

    counts = {"colliders": 0, "rigid_bodies": 0, "joints": 0}
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path != root_path and not path.startswith(f"{root_path}/"):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            counts["colliders"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            counts["rigid_bodies"] += 1
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            counts["joints"] += 1
    return counts


def _validate_alignment(
    stage,
    *,
    source_frame_path: str,
    target_frame_path: str,
    expected_yaw_deg: float,
    Usd,
    UsdGeom,
) -> dict[str, Any]:
    import math

    source_frame = stage.GetPrimAtPath(source_frame_path)
    target_frame = stage.GetPrimAtPath(target_frame_path)
    if not source_frame.IsValid() or not target_frame.IsValid():
        raise VisualRoomError("doorframe disappeared while aligning the visual room")
    cache = UsdGeom.XformCache()
    source = cache.GetLocalToWorldTransform(source_frame).RemoveScaleShear()
    target = cache.GetLocalToWorldTransform(target_frame).RemoveScaleShear()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    source_range = bounds.ComputeWorldBound(source_frame).ComputeAlignedRange()
    target_range = bounds.ComputeWorldBound(target_frame).ComputeAlignedRange()
    center_delta = source_range.GetMidpoint() - target_range.GetMidpoint()
    center_error = center_delta.GetLength()
    if center_error > 1e-4:
        raise VisualRoomError(
            f"visual doorway center error is {center_error:.6f} m"
        )
    size_delta = source_range.GetSize() - target_range.GetSize()
    size_error = size_delta.GetLength()
    if size_error > 1e-4:
        raise VisualRoomError(f"visual doorway size error is {size_error:.6f} m")
    relative = source.ExtractRotation() * target.ExtractRotation().GetInverse()
    yaw_error_deg = abs(relative.GetAngle() - abs(expected_yaw_deg))
    if not math.isfinite(yaw_error_deg) or yaw_error_deg > 1e-3:
        raise VisualRoomError(
            f"visual doorway yaw is {relative.GetAngle():.6f} deg; "
            f"expected {expected_yaw_deg:.6f} deg"
        )
    return {
        "basis": "doorframe_world_aligned_bounds",
        "center_error_m": center_error,
        "size_error_m": size_error,
        "relative_yaw_deg": relative.GetAngle(),
        "source_center_w": list(source_range.GetMidpoint()),
        "target_center_w": list(target_range.GetMidpoint()),
        "source_size_m": list(source_range.GetSize()),
        "target_size_m": list(target_range.GetSize()),
        "source_frame_origin_w": list(source.ExtractTranslation()),
        "target_frame_origin_w": list(target.ExtractTranslation()),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "DEFAULT_VISUAL_ROOM_PROFILE",
    "VISUAL_ROOM_PROFILE_NAMES",
    "VISUAL_ROOM_RENDER_WARMUP_FRAMES",
    "VisualRoomError",
    "VisualRoomProfile",
    "attach_visual_room",
    "visual_room_profile",
]
