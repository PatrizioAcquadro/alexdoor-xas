"""Pure rollout-video and presentation-room contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.eval.video import (
    RolloutVideoRecorder,
    VideoCaptureError,
    resolve_video_output,
)
from alexdoor_xas.eval.visual_room import (
    DEFAULT_VISUAL_ROOM_PROFILE,
    VISUAL_ROOM_PROFILE_NAMES,
    VISUAL_ROOM_RENDER_WARMUP_FRAMES,
    VisualRoomError,
    visual_room_profile,
)

# --- test_video ---


class RenderEnv:
    def __init__(self, frame):
        self.frame = frame

    def render(self):
        return self.frame


def test_video_output_is_confined_to_outputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outputs = repo / "outputs"
    path = resolve_video_output(
        "outputs/demo/rollout.mp4",
        repo_root=repo,
        outputs_root=outputs,
    )

    assert path == outputs / "demo" / "rollout.mp4"

    with pytest.raises(VideoCaptureError, match="must be under"):
        resolve_video_output(
            "../outside.mp4",
            repo_root=repo,
            outputs_root=outputs,
        )
    with pytest.raises(VideoCaptureError, match=r"end in \.mp4"):
        resolve_video_output(
            "outputs/demo/rollout.avi",
            repo_root=repo,
            outputs_root=outputs,
        )


def test_video_output_refuses_existing_video_or_sidecar(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    video = outputs / "demo.mp4"
    video.parent.mkdir()
    video.with_suffix(".json").write_text("{}\n")

    with pytest.raises(VideoCaptureError, match="refusing to overwrite"):
        resolve_video_output(video, repo_root=tmp_path, outputs_root=outputs)


def test_recorder_captures_rgb_copy() -> None:
    frame = np.zeros((4, 6, 4), dtype=np.uint8)
    recorder = RolloutVideoRecorder(Path("rollout.mp4"), fps=60)

    recorder.capture(RenderEnv(frame), tick=1)
    frame[:] = 255

    assert recorder.frame_count == 1
    assert recorder._frames[0].shape == (4, 6, 3)
    assert not recorder._frames[0].any()


def test_presentation_frames_trim_blank_and_add_holds() -> None:
    recorder = RolloutVideoRecorder(
        Path("rollout.mp4"),
        fps=2,
        capture_fps=4,
        intro_hold_s=1.0,
        outro_hold_s=1.5,
    )
    blank = np.zeros((2, 3, 3), dtype=np.uint8)
    first = np.full((2, 3, 3), 32, dtype=np.uint8)
    last = np.full((2, 3, 3), 64, dtype=np.uint8)
    for tick, frame in enumerate((blank, first, last), start=1):
        recorder.capture(RenderEnv(frame), tick=tick)

    frames, dropped, intro, outro = recorder._presentation_frames()

    assert dropped == 1
    assert intro == 2
    assert outro == 3
    assert len(frames) == 7
    assert np.array_equal(frames[0], first)
    assert np.array_equal(frames[-1], last)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fps": 0}, "playback fps"),
        ({"fps": 30, "capture_fps": 0}, "capture fps"),
        ({"fps": 30, "intro_hold_s": -1.0}, "holds"),
        ({"fps": 30, "outro_hold_s": float("nan")}, "holds"),
    ],
)
def test_recorder_rejects_invalid_playback_settings(kwargs, message: str) -> None:
    with pytest.raises(VideoCaptureError, match=message):
        RolloutVideoRecorder(Path("rollout.mp4"), **kwargs)


@pytest.mark.parametrize(
    "frame, message",
    [
        (None, "returned no frame"),
        (np.zeros((4, 6), dtype=np.uint8), "must have shape"),
        (np.zeros((4, 6, 3), dtype=np.float32), "must use uint8"),
    ],
)
def test_recorder_rejects_invalid_frames(frame, message: str) -> None:
    recorder = RolloutVideoRecorder(Path("rollout.mp4"), fps=60)

    with pytest.raises(VideoCaptureError, match=message):
        recorder.capture(RenderEnv(frame), tick=1)


# --- test_visual_room ---


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
