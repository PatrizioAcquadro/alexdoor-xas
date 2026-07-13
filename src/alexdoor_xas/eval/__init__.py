"""Evaluation role: metrics, failure labels, plots, and run reports."""

from __future__ import annotations

from .failures import FAILURE_LABELS, label_episode
from .metrics import aggregate_metrics, episode_metrics
from .sanity import (
    FORCE_DATASET_LIMIT_N,
    FORCE_WARN_N,
    SanityResult,
    check_alex_episode,
    contact_force_diagnostics,
)

__all__ = [
    "FAILURE_LABELS",
    "FORCE_DATASET_LIMIT_N",
    "FORCE_WARN_N",
    "SanityResult",
    "aggregate_metrics",
    "check_alex_episode",
    "contact_force_diagnostics",
    "episode_metrics",
    "label_episode",
]
