"""Configuration shared by learned policy families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA


@dataclass(frozen=True)
class PolicyDatasetCfg:
    task: str = "door_push_alex_v2"
    space: str = A2_EE_DELTA
    version: str = "v2_pose"
    view_id: str | None = None
    obs_preset: str = "core"

    @property
    def dataset_dir(self) -> Path:
        return paths.DATASETS_DIR / self.task / self.space / self.version


@dataclass(frozen=True)
class PolicyRunCfg:
    output_root: str | None = None
