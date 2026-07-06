"""Optional experiment tracking helpers.

The package is safe to import without W&B installed. The SDK is imported lazily
only when a run is explicitly started in ``offline`` or ``online`` mode.
"""

from .wandb import (
    NoOpWandbRun,
    WandbConfig,
    WandbRun,
    load_wandb_config,
    sanitize_wandb_config,
    start_wandb_run,
)

__all__ = [
    "NoOpWandbRun",
    "WandbConfig",
    "WandbRun",
    "load_wandb_config",
    "sanitize_wandb_config",
    "start_wandb_run",
]
