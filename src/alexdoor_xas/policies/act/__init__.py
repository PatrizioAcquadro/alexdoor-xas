"""ACT-style action-chunk imitation baseline (Phase 3.2).

Only the torch-free config surface is re-exported here so the package can be
imported before Isaac AppLauncher / torch initialization. Model, training,
checkpoint, and policy-wrapper modules import torch and are imported directly
(``alexdoor_xas.policies.act.model`` etc.) by training/eval code.
"""

from .config import (
    ActConfig,
    ActConfigError,
    ActDatasetCfg,
    ActModelCfg,
    ActRolloutCfg,
    ActRunCfg,
    ActTrainCfg,
    act_config_from_dict,
    load_act_config,
)

__all__ = [
    "ActConfig",
    "ActConfigError",
    "ActDatasetCfg",
    "ActModelCfg",
    "ActRolloutCfg",
    "ActRunCfg",
    "ActTrainCfg",
    "act_config_from_dict",
    "load_act_config",
]
