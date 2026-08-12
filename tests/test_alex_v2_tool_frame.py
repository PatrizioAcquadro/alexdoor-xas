"""Collision-derived Alex V2 tool-frame tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest
from alexdoor_xas.assets.alex_v2_tool_frame import (
    ToolFrameError,
    derive_right_gripper_tool_frame,
)


def test_tool_frame_matches_the_current_collision_union() -> None:
    manifest = build_alex_v2_manifest()
    first = derive_right_gripper_tool_frame(manifest, (2.0, 0.0, 0.0))
    second = derive_right_gripper_tool_frame(manifest, (1.0, 0.0, 0.0))

    assert first == second
    assert first.to_dict() == {
        "parent_link": "RIGHT_GRIPPER_Z_LINK",
        "translation_m": pytest.approx([0.1179961994235339, 0.0, -0.06250775384897367]),
        "orientation_xyzw": pytest.approx([0.0, 0.0, 0.0, 1.0]),
        "contact_normal_link": [1.0, 0.0, 0.0],
    }


def test_tool_frame_rejects_invalid_collision_geometry() -> None:
    manifest = deepcopy(build_alex_v2_manifest())
    manifest["collision_profile"]["links"]["RIGHT_GRIPPER_Z_LINK"][0]["shape"] = "mesh"

    with pytest.raises(ToolFrameError, match="refusing an approximate"):
        derive_right_gripper_tool_frame(manifest, (1.0, 0.0, 0.0))
    with pytest.raises(ToolFrameError, match="missing right-gripper"):
        derive_right_gripper_tool_frame({}, (1.0, 0.0, 0.0))
