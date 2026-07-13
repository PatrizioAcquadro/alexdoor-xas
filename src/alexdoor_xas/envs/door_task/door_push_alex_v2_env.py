"""Registered production executor for the calibrated ``door_push_alex_v2`` task."""

from __future__ import annotations

from pathlib import Path

from alexdoor_xas.assets.alex_v2 import (
    build_alex_v2_door_asset,
)
from alexdoor_xas.calibration.alex_v2_door import (
    load_validated_alex_v2_door_calibration,
)

from .alex_v2_runtime import require_current_collision_tool_frame
from .door_push_alex_v2_env_cfg import DoorPushAlexV2EnvCfg
from .door_push_alex_v2_executor import DoorPushAlexV2Executor


class AlexV2TaskNotReadyError(RuntimeError):
    """Raised when the exact V2 runtime cannot be constructed safely."""


class DoorPushAlexV2Env(DoorPushAlexV2Executor):
    """Production V2 task; only fully gated calibration reaches execution."""

    cfg: DoorPushAlexV2EnvCfg
    candidate_only = False

    def __init__(
        self,
        cfg: DoorPushAlexV2EnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        asset, runtime_asset = build_alex_v2_door_asset()
        calibration = load_validated_alex_v2_door_calibration(
            cfg.calibration_path,
            runtime_asset=runtime_asset,
            runtime_versions=_runtime_versions(),
        )
        require_current_collision_tool_frame(asset.manifest, calibration)
        super().__init__(
            cfg,
            calibration=calibration,
            runtime_asset=runtime_asset,
            runtime_manifest=asset.manifest,
            render_mode=render_mode,
            **kwargs,
        )


def _runtime_versions() -> dict[str, str]:
    return {
        "isaac_sim": _version(Path.home() / "isaacsim" / "VERSION"),
        "isaac_lab": _version(Path.home() / "IsaacLab" / "VERSION"),
    }


def _version(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise AlexV2TaskNotReadyError(
            f"runtime version file is unavailable: {path}"
        ) from error
    if not value:
        raise AlexV2TaskNotReadyError(f"runtime version file is empty: {path}")
    return value


__all__ = ["AlexV2TaskNotReadyError", "DoorPushAlexV2Env"]
