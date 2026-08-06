"""Pure contracts for presentation-only combined-scene room capture."""

from __future__ import annotations

import pytest

from alexdoor_xas.eval.visual_room import (
    DEFAULT_VISUAL_ROOM_PROFILE,
    VISUAL_ROOM_PROFILE_NAMES,
    VISUAL_ROOM_RENDER_WARMUP_FRAMES,
    VisualRoomError,
    visual_room_profile,
)


def test_living_room_profile_is_frozen_for_video_capture() -> None:
    assert VISUAL_ROOM_PROFILE_NAMES == ("floorplan212_living_room",)
    assert DEFAULT_VISUAL_ROOM_PROFILE == "floorplan212_living_room"
    assert VISUAL_ROOM_RENDER_WARMUP_FRAMES == 8

    profile = visual_room_profile(DEFAULT_VISUAL_ROOM_PROFILE)

    assert profile.label == "FloorPlan212 living room"
    assert profile.room_source_prim == "/World/scene_01"
    assert profile.hallway_source_prim == "/World/Hallway"
    assert profile.source_door_subpath == "Door"
    assert profile.context_yaw_deg == pytest.approx(180.0)
    assert profile.camera_eye == pytest.approx((1.85, 1.15, 2.05))
    assert profile.camera_lookat == pytest.approx((-0.70, -0.05, 0.92))
    assert profile.ambient_light_intensity == pytest.approx(450.0)


def test_visual_room_profile_rejects_unknown_names() -> None:
    with pytest.raises(VisualRoomError, match="unknown visual room"):
        visual_room_profile("fallback_room")
