"""Collision-derived contact frame for the official Alex V2 right gripper."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_GRIPPER_LINK = "RIGHT_GRIPPER_Z_LINK"
_SUPPORTED_SHAPES = frozenset({"box", "capsule", "cylinder", "sphere"})


class ToolFrameError(ValueError):
    """Raised when collision geometry cannot define an exact tool frame."""


@dataclass(frozen=True)
class ToolFrame:
    parent_link: str
    translation_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    contact_normal_link: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_link": self.parent_link,
            "translation_m": list(self.translation_m),
            "orientation_xyzw": list(self.orientation_xyzw),
            "contact_normal_link": list(self.contact_normal_link),
        }


def derive_right_gripper_tool_frame(
    manifest: Mapping[str, Any],
    door_contact_normal_link: Sequence[float],
) -> ToolFrame:
    """Return the support point of the official collision union along a normal."""

    collisions = _right_gripper_collisions(manifest)
    normal = _normalize(_vec3(door_contact_normal_link, "door contact normal"))
    candidates: list[tuple[float, str, tuple[float, float, float]]] = []
    for record in collisions:
        shape = str(record.get("shape", ""))
        if shape not in _SUPPORTED_SHAPES:
            raise ToolFrameError(
                f"unsupported right-gripper collision shape {shape!r}; "
                "refusing an approximate tool frame"
            )
        name = str(record.get("name", ""))
        if not name:
            raise ToolFrameError("right-gripper collision is missing its name")
        origin = _mapping(record.get("origin"), f"collision {name} origin")
        center = _vec3(origin.get("xyz_m"), f"collision {name} xyz_m")
        rpy = _vec3(origin.get("rpy_rad"), f"collision {name} rpy_rad")
        rotation = _rpy_matrix(rpy)
        local_normal = _mat_t_vec(rotation, normal)
        local_support = _primitive_support(
            shape,
            _mapping(record.get("dimensions"), f"collision {name} dimensions"),
            local_normal,
        )
        point = _add(center, _mat_vec(rotation, local_support))
        candidates.append((_dot(normal, point), name, point))
    if not candidates:
        raise ToolFrameError("manifest has no right-gripper collision primitives")
    _, _, point = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    orientation = _frame_quaternion(normal)
    return ToolFrame(
        parent_link=_GRIPPER_LINK,
        translation_m=point,
        orientation_xyzw=orientation,
        contact_normal_link=normal,
    )


def _right_gripper_collisions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = manifest.get("collision_profile")
    links = profile.get("links") if isinstance(profile, Mapping) else None
    values = links.get(_GRIPPER_LINK) if isinstance(links, Mapping) else None
    if not isinstance(values, list):
        raise ToolFrameError("manifest is missing right-gripper collision records")
    records = [_mapping(value, "right-gripper collision") for value in values]
    for record in records:
        if record.get("link") != _GRIPPER_LINK:
            raise ToolFrameError("collision record is not attached to the right gripper")
    return sorted((dict(record) for record in records), key=lambda item: str(item["name"]))


def _primitive_support(
    shape: str,
    dimensions: Mapping[str, Any],
    direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    direction = _normalize(direction)
    if shape == "sphere":
        return _scale(direction, _positive(dimensions, "radius_m"))
    if shape == "box":
        size = _vec3(dimensions.get("size_m"), "box size_m")
        if any(value <= 0.0 for value in size):
            raise ToolFrameError("box dimensions must be positive")
        return tuple(
            math.copysign(value / 2.0, axis) for value, axis in zip(size, direction, strict=True)
        )
    radius = _positive(dimensions, "radius_m")
    length = _positive(dimensions, "length_m")
    radial_norm = math.hypot(direction[0], direction[1])
    radial = (
        (radius * direction[0] / radial_norm, radius * direction[1] / radial_norm)
        if radial_norm > 1e-15
        else (0.0, 0.0)
    )
    half_axis = math.copysign(length / 2.0, direction[2])
    cylinder = (radial[0], radial[1], half_axis)
    if shape == "cylinder":
        return cylinder
    return _add(cylinder, _scale(direction, radius))


def _frame_quaternion(normal: tuple[float, float, float]) -> tuple[float, float, float, float]:
    hint = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.95 else (0.0, 1.0, 0.0)
    y_axis = _normalize(_cross(hint, normal))
    z_axis = _cross(normal, y_axis)
    matrix = (
        (normal[0], y_axis[0], z_axis[0]),
        (normal[1], y_axis[1], z_axis[1]),
        (normal[2], y_axis[2], z_axis[2]),
    )
    return _matrix_quaternion(matrix)


def _rpy_matrix(rpy: tuple[float, float, float]):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_quaternion(matrix) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        nxt = (index + 1) % 3
        last = (index + 2) % 3
        scale = math.sqrt(1.0 + matrix[index][index] - matrix[nxt][nxt] - matrix[last][last]) * 2
        values = [0.0, 0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[3] = (matrix[last][nxt] - matrix[nxt][last]) / scale
        values[nxt] = (matrix[nxt][index] + matrix[index][nxt]) / scale
        values[last] = (matrix[last][index] + matrix[index][last]) / scale
        quat = tuple(values)
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(value / norm for value in quat)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolFrameError(f"{label} must be a mapping")
    return value


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ToolFrameError(f"{label} must contain exactly three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ToolFrameError(f"{label} must be finite")
    return result


def _positive(value: Mapping[str, Any], key: str) -> float:
    result = float(value.get(key, 0.0))
    if not math.isfinite(result) or result <= 0.0:
        raise ToolFrameError(f"{key} must be positive")
    return result


def _normalize(value):
    norm = math.sqrt(_dot(value, value))
    if norm <= 1e-15:
        raise ToolFrameError("direction vector must be non-zero")
    return _scale(value, 1.0 / norm)


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _scale(value, scalar):
    return tuple(scalar * item for item in value)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _mat_vec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _mat_t_vec(matrix, vector):
    return tuple(sum(matrix[row][col] * vector[row] for row in range(3)) for col in range(3))


__all__ = ["ToolFrame", "ToolFrameError", "derive_right_gripper_tool_frame"]
