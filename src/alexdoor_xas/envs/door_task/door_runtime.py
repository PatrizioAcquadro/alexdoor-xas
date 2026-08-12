"""Runtime helpers shared by every articulated door benchmark environment."""

from __future__ import annotations

DOORFRAME_PRIM_TEMPLATE = "/World/envs/env_{index}/DoorTaskScene/DoorTaskDoor/Doorframe"


def resolve_hinge_joint_id(joint_names: list[str], hinge_joint_name: str) -> int:
    """Return the index of the unique requested hinge joint."""
    target = hinge_joint_name.lower()
    matches = [
        idx
        for idx, name in enumerate(joint_names)
        if name.lower() == target or name.rsplit("/", 1)[-1].lower() == target
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches and len(joint_names) == 1:
        return 0
    raise RuntimeError(
        f"could not identify one hinge joint named {hinge_joint_name!r}; "
        f"available joints: {joint_names}"
    )


def read_doorframe_from_stage(env_id: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return one environment's world-space door-frame pose as ``(pos, quat_xyzw)``."""
    import omni.usd  # noqa: PLC0415 - Kit runtime import, valid after AppLauncher.
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_path = DOORFRAME_PRIM_TEMPLATE.format(index=env_id)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"door frame prim not found on stage: {prim_path}")
    transform = cache.GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    quat = transform.RemoveScaleShear().ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        (translation[0], translation[1], translation[2]),
        (imaginary[0], imaginary[1], imaginary[2], quat.GetReal()),
    )


__all__ = ["DOORFRAME_PRIM_TEMPLATE", "read_doorframe_from_stage", "resolve_hinge_joint_id"]
