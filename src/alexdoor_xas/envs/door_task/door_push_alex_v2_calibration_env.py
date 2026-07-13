"""Unregistered candidate-only Alex V2 calibration executor.

This module is intentionally absent from Gym registration and exports.  A
caller must import the class explicitly and provide a candidate calibration
path, making it impossible to select through the production task ID.
"""

from __future__ import annotations

from pathlib import Path

from alexdoor_xas.assets.alex_v2 import (
    build_alex_v2_door_asset,
)
from alexdoor_xas.calibration.alex_v2_door import (
    load_candidate_alex_v2_door_calibration,
)

from .alex_v2_runtime import require_current_collision_tool_frame
from .door_push_alex_v2_env import _runtime_versions
from .door_push_alex_v2_env_cfg import DoorPushAlexV2EnvCfg
from .door_push_alex_v2_executor import DoorPushAlexV2Executor


class DoorPushAlexV2CalibrationEnv(DoorPushAlexV2Executor):
    """Execute structurally valid candidate calibration outside Gym only."""

    candidate_only = True
    gym_registration_allowed = False

    def __init__(
        self,
        cfg: DoorPushAlexV2EnvCfg,
        *,
        candidate_calibration_path: str | Path,
        render_mode: str | None = None,
        **kwargs,
    ):
        asset, runtime_asset = build_alex_v2_door_asset()
        calibration = load_candidate_alex_v2_door_calibration(
            candidate_calibration_path,
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


# Deliberately empty: candidate execution is never re-exported from the task
# package and no module-level factory is provided for Gym registration.
__all__: list[str] = []
