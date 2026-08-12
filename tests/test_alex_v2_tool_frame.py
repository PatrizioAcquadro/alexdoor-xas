"""Pure tests for collision-derived Alex V2 tool geometry."""

from __future__ import annotations

import pytest

from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest
from alexdoor_xas.assets.alex_v2_tool_frame import (
    ToolFrameError,
    derive_right_gripper_tool_frame,
)


def _collision(name, shape, xyz, dimensions, rpy=(0.0, 0.0, 0.0)):
    return {
        "name": name,
        "link": "RIGHT_GRIPPER_Z_LINK",
        "origin": {"xyz_m": list(xyz), "rpy_rad": list(rpy)},
        "shape": shape,
        "dimensions": dimensions,
    }


def _manifest():
    collisions = [
        _collision(
            "right_gripper_z_collision",
            "capsule",
            (0.0, 0.01, -0.1),
            {"radius_m": 0.04, "length_m": 0.1},
        ),
        _collision(
            "right_fist_collision",
            "cylinder",
            (0.0, 0.012, -0.075),
            {"radius_m": 0.047, "length_m": 0.06},
        ),
        _collision(
            "right_finger_collision",
            "box",
            (-0.001, -0.013, -0.145),
            {"size_m": [0.08, 0.025, 0.075]},
        ),
        _collision(
            "right_thumb_collision",
            "cylinder",
            (0.08, 0.0, -0.06),
            {"radius_m": 0.015, "length_m": 0.07},
            rpy=(0.0, -1.1, 0.0),
        ),
    ]
    return {
        "contact_support_geometry": {
            "sides": {
                "right": {
                    "link": "RIGHT_GRIPPER_Z_LINK",
                    "collisions": collisions,
                }
            }
        }
    }


def test_tool_frame_uses_exact_collision_union_support() -> None:
    tool = derive_right_gripper_tool_frame(_manifest(), (1.0, 0.0, 0.0))

    assert tool.parent_link == "RIGHT_GRIPPER_Z_LINK"
    assert tool.support_shape == "right_thumb_collision"
    assert tool.translation_m[0] == pytest.approx(tool.support_distance_m)
    assert tool.support_distance_m > 0.10
    assert len(tool.collision_union_sha256) == 64
    assert sum(value * value for value in tool.orientation_xyzw) == pytest.approx(1.0)


def test_tool_frame_is_deterministic_and_normal_is_normalized() -> None:
    first = derive_right_gripper_tool_frame(_manifest(), (2.0, 0.0, 0.0))
    second = derive_right_gripper_tool_frame(_manifest(), (1.0, 0.0, 0.0))
    assert first == second


def test_tool_frame_consumes_the_static_urdf_manifest_schema() -> None:
    tool = derive_right_gripper_tool_frame(build_alex_v2_manifest(), (1.0, 0.0, 0.0))

    assert tool.parent_link == "RIGHT_GRIPPER_Z_LINK"
    assert tool.support_shape == "right_thumb_collision"
    assert tool.support_distance_m > 0.10


def test_tool_frame_rejects_unsupported_or_missing_collision_geometry() -> None:
    manifest = _manifest()
    manifest["contact_support_geometry"]["sides"]["right"]["collisions"][0]["shape"] = "mesh"
    with pytest.raises(ToolFrameError, match="refusing an approximate"):
        derive_right_gripper_tool_frame(manifest, (1.0, 0.0, 0.0))

    with pytest.raises(ToolFrameError, match="missing right-gripper"):
        derive_right_gripper_tool_frame({}, (1.0, 0.0, 0.0))
