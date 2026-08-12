"""Evaluation role: factual metrics, plots, and run reports."""

from __future__ import annotations

from .metrics import aggregate_metrics, episode_metrics
from .sanity import (
    FORCE_DATASET_LIMIT_N,
    FORCE_WARN_N,
    SanityResult,
    check_alex_episode,
    contact_force_diagnostics,
)

__all__ = [
    "FORCE_DATASET_LIMIT_N",
    "FORCE_WARN_N",
    "SanityResult",
    "aggregate_metrics",
    "check_alex_episode",
    "contact_force_diagnostics",
    "episode_metrics",
]
