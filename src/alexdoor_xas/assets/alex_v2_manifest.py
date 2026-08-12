"""Build the Alex V2 identity manifest directly from its pinned static URDF."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from alexdoor_xas import paths

ALEX_V2_STANDARD_PROFILE = "alex_v2_fullbody_standard_no_external_hands"
STATIC_MANIFEST_KIND = "alexdoor.alex_v2.static.v1"
EXPECTED_ALEX_V2_URDF_SHA256 = "7742b88d9cb81e80f3d1e5c1906e31f38ca03734085454505e550b24009920b3"
EXPECTED_MOVABLE_JOINT_COUNT = 29
EXPECTED_COLLISION_RECORD_COUNT = 32
SUPPORTED_COLLISION_SHAPES = frozenset({"box", "capsule", "cylinder", "sphere"})


class AlexV2ManifestError(ValueError):
    """Raised when the static Alex V2 URDF violates its frozen contract."""


def build_alex_v2_manifest(urdf_path: str | Path | None = None) -> dict[str, Any]:
    """Parse the pinned standard URDF into the pure-Python asset manifest."""
    path = Path(urdf_path) if urdf_path is not None else paths.ALEX_V2_URDF
    path = path.expanduser().resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AlexV2ManifestError(f"cannot read Alex V2 URDF {path}: {error}") from error
    urdf_sha256 = hashlib.sha256(payload).hexdigest()
    if urdf_sha256 != EXPECTED_ALEX_V2_URDF_SHA256:
        raise AlexV2ManifestError(
            "Alex V2 URDF identity differs from the frozen standard asset: "
            f"expected {EXPECTED_ALEX_V2_URDF_SHA256}, got {urdf_sha256}"
        )

    try:
        robot = ET.fromstring(payload)
    except ET.ParseError as error:
        raise AlexV2ManifestError(f"cannot parse Alex V2 URDF {path}: {error}") from error
    if robot.tag != "robot" or robot.get("name") != "AlexV2":
        raise AlexV2ManifestError(f"unexpected Alex V2 URDF root in {path}")

    movable_joints = _movable_joint_names(robot)
    collision_profile = _collision_profile(robot)
    core: dict[str, Any] = {
        "kind": STATIC_MANIFEST_KIND,
        "profile": ALEX_V2_STANDARD_PROFILE,
        "urdf_sha256": urdf_sha256,
        "movable_joint_count": len(movable_joints),
        "movable_joints": movable_joints,
        "collision_profile": collision_profile,
    }
    fingerprint = _canonical_sha256(core)
    asset_id = f"{ALEX_V2_STANDARD_PROFILE}:{urdf_sha256}"
    return {
        **core,
        "asset_id": asset_id,
        "robot_asset_id": asset_id,
        "robot_asset_sha256": urdf_sha256,
        "fingerprint": fingerprint,
    }


def _movable_joint_names(robot: ET.Element) -> list[str]:
    names: list[str] = []
    for joint in robot.findall("joint"):
        joint_type = joint.get("type", "")
        if not joint_type:
            raise AlexV2ManifestError("Alex V2 joint is missing its type")
        if joint_type == "fixed":
            continue
        name = joint.get("name", "")
        if not name:
            raise AlexV2ManifestError("Alex V2 movable joint is missing its name")
        names.append(name)
    if len(names) != EXPECTED_MOVABLE_JOINT_COUNT:
        raise AlexV2ManifestError(
            "Alex V2 movable-joint count drifted: "
            f"expected {EXPECTED_MOVABLE_JOINT_COUNT}, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise AlexV2ManifestError("Alex V2 movable-joint names must be unique")
    return names


def _collision_profile(robot: ET.Element) -> dict[str, Any]:
    links: dict[str, list[dict[str, Any]]] = {}
    for link in robot.findall("link"):
        records = [
            _collision_record(link.get("name", ""), collision)
            for collision in link.findall("collision")
        ]
        if records:
            links[link.get("name", "")] = records
    record_count = sum(len(records) for records in links.values())
    if record_count != EXPECTED_COLLISION_RECORD_COUNT:
        raise AlexV2ManifestError(
            "Alex V2 primitive-collision record count drifted: "
            f"expected {EXPECTED_COLLISION_RECORD_COUNT}, got {record_count}"
        )
    return {"links": links}


def _collision_record(link_name: str, collision: ET.Element) -> dict[str, Any]:
    if not link_name:
        raise AlexV2ManifestError("Alex V2 collision is attached to an unnamed link")
    name = collision.get("name", "")
    if not name:
        raise AlexV2ManifestError(f"Alex V2 collision on {link_name} is missing its name")
    geometry = collision.find("geometry")
    if geometry is None:
        raise AlexV2ManifestError(f"Alex V2 collision {name!r} is missing geometry")
    shapes = [child for child in geometry if child.tag in SUPPORTED_COLLISION_SHAPES]
    if len(shapes) != 1 or len(geometry) != 1:
        tags = [child.tag for child in geometry]
        raise AlexV2ManifestError(
            f"Alex V2 collision {name!r} must contain one supported primitive; got {tags}"
        )
    shape = shapes[0]
    origin = collision.find("origin")
    return {
        "name": name,
        "link": link_name,
        "shape": shape.tag,
        "origin": {
            "xyz_m": _vector(origin.get("xyz") if origin is not None else None, name),
            "rpy_rad": _vector(origin.get("rpy") if origin is not None else None, name),
        },
        "dimensions": _dimensions(shape, name),
    }


def _vector(text: str | None, collision_name: str) -> list[float]:
    try:
        values = [float(item) for item in (text or "0 0 0").split()]
    except ValueError as error:
        raise AlexV2ManifestError(
            f"Alex V2 collision {collision_name!r} has a non-numeric origin"
        ) from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise AlexV2ManifestError(
            f"Alex V2 collision {collision_name!r} origin must be a finite three-vector"
        )
    return values


def _dimensions(shape: ET.Element, collision_name: str) -> dict[str, Any]:
    if shape.tag == "box":
        size = _vector(shape.get("size"), collision_name)
        if any(value <= 0.0 for value in size):
            raise AlexV2ManifestError(
                f"Alex V2 collision {collision_name!r} box size must be positive"
            )
        return {"size_m": size}
    radius = _positive_float(shape.get("radius"), collision_name, "radius")
    if shape.tag == "sphere":
        return {"radius_m": radius}
    length = _positive_float(shape.get("length"), collision_name, "length")
    return {"radius_m": radius, "length_m": length}


def _positive_float(value: str | None, collision_name: str, label: str) -> float:
    try:
        number = float(value or "")
    except ValueError as error:
        raise AlexV2ManifestError(
            f"Alex V2 collision {collision_name!r} has an invalid {label}"
        ) from error
    if not math.isfinite(number) or number <= 0.0:
        raise AlexV2ManifestError(
            f"Alex V2 collision {collision_name!r} {label} must be finite and positive"
        )
    return number


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ALEX_V2_STANDARD_PROFILE",
    "EXPECTED_ALEX_V2_URDF_SHA256",
    "EXPECTED_COLLISION_RECORD_COUNT",
    "EXPECTED_MOVABLE_JOINT_COUNT",
    "STATIC_MANIFEST_KIND",
    "AlexV2ManifestError",
    "build_alex_v2_manifest",
]
