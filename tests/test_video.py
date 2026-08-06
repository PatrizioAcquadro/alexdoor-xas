"""Pure contract tests for fail-closed rollout-video capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.eval.video import (
    RolloutVideoRecorder,
    VideoCaptureError,
    resolve_video_output,
)


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
