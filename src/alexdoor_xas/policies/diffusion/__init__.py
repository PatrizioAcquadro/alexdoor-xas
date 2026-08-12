"""Diffusion Policy baseline (Phase 3.3).

Only the torch-free/diffusers-free config surface is re-exported here so the
package can be imported before AppLauncher/torch initialization. The model,
trainer, checkpoint, and rollout policy are imported by their full module
paths from scripts (they require torch and diffusers).
"""

from alexdoor_xas.policies.diffusion.config import (
    DiffusionConfig,
    DiffusionConfigError,
    DiffusionDatasetCfg,
    DiffusionModelCfg,
    DiffusionRolloutCfg,
    DiffusionRunCfg,
    DiffusionTrainCfg,
    diffusion_config_from_dict,
    load_diffusion_config,
)

__all__ = [
    "DiffusionConfig",
    "DiffusionConfigError",
    "DiffusionDatasetCfg",
    "DiffusionModelCfg",
    "DiffusionRolloutCfg",
    "DiffusionRunCfg",
    "DiffusionTrainCfg",
    "diffusion_config_from_dict",
    "load_diffusion_config",
]
