"""Fail-closed RGB rollout-video capture for simulator evaluations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class VideoCaptureError(RuntimeError):
    """A requested rollout video could not be captured or encoded safely."""


def resolve_video_output(
    value: str | Path,
    *,
    repo_root: Path,
    outputs_root: Path,
) -> Path:
    """Resolve one new MP4 path and confine it to the repository outputs tree."""
    raw_path = Path(value)
    candidate = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(outputs_root.resolve())
    except ValueError as error:
        raise VideoCaptureError(
            f"video output must be under {outputs_root}, got {resolved}"
        ) from error
    if resolved.suffix.lower() != ".mp4":
        raise VideoCaptureError(f"video output must end in .mp4, got {resolved}")
    sidecar = resolved.with_suffix(".json")
    existing = [path for path in (resolved, sidecar) if path.exists()]
    if existing:
        raise VideoCaptureError(
            "refusing to overwrite existing video artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    return resolved


@dataclass
class RolloutVideoRecorder:
    """Collect fixed-shape RGB frames and atomically encode one MP4."""

    output_path: Path
    fps: int
    _frames: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _frame_shape: tuple[int, int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        if self.output_path.suffix.lower() != ".mp4":
            raise VideoCaptureError("rollout video output must use the .mp4 extension")
        if self.fps <= 0:
            raise VideoCaptureError("rollout video fps must be positive")

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def capture(self, env: Any, tick: int) -> None:
        """Render and retain the post-action RGB frame for one completed tick."""
        try:
            frame = env.render()
        except Exception as error:  # noqa: BLE001 - normalize the simulator boundary.
            raise VideoCaptureError(f"render failed after tick {tick}: {error}") from error
        if frame is None:
            raise VideoCaptureError(
                "render returned no frame; launch with --enable_cameras and "
                "render_mode='rgb_array'"
            )
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise VideoCaptureError(
                f"rendered frame must have shape (H, W, 3|4), got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise VideoCaptureError(
                f"rendered frame must use uint8 RGB values, got {array.dtype}"
            )
        rgb = np.ascontiguousarray(array[:, :, :3])
        if self._frame_shape is None:
            self._frame_shape = rgb.shape
        elif rgb.shape != self._frame_shape:
            raise VideoCaptureError(
                f"rendered frame shape changed from {self._frame_shape} to {rgb.shape}"
            )
        self._frames.append(rgb.copy())

    def write(self) -> dict[str, Any]:
        """Encode all captured frames and return auditable file metadata."""
        if not self._frames or self._frame_shape is None:
            raise VideoCaptureError("cannot encode a rollout video with no frames")
        if self.output_path.exists():
            raise VideoCaptureError(f"refusing to overwrite existing {self.output_path}")

        import imageio.v3 as iio

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(
            f".{self.output_path.stem}.encoding{self.output_path.suffix}"
        )
        if temporary.exists():
            raise VideoCaptureError(f"temporary video path already exists: {temporary}")
        try:
            iio.imwrite(temporary, np.stack(self._frames), fps=self.fps)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise VideoCaptureError("video encoder produced no file content")
            temporary.replace(self.output_path)
        except Exception as error:  # noqa: BLE001 - include encoder/plugin failures.
            raise VideoCaptureError(f"video encode failed: {error}") from error
        finally:
            if temporary.exists():
                temporary.unlink()

        height, width, channels = self._frame_shape
        return {
            "path": str(self.output_path),
            "sha256": hashlib.sha256(self.output_path.read_bytes()).hexdigest(),
            "size_bytes": self.output_path.stat().st_size,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": width,
            "height": height,
            "channels": channels,
            "duration_s": self.frame_count / self.fps,
        }


__all__ = [
    "RolloutVideoRecorder",
    "VideoCaptureError",
    "resolve_video_output",
]
