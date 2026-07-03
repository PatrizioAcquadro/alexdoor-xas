"""Evaluation role: metrics, failure labels, plots, and run reports."""

from __future__ import annotations

from .failures import FAILURE_LABELS, label_episode
from .metrics import aggregate_metrics, episode_metrics
from .sanity import SanityResult, check_alex_episode

__all__ = [
    "FAILURE_LABELS",
    "SanityResult",
    "aggregate_metrics",
    "check_alex_episode",
    "episode_metrics",
    "label_episode",
]
