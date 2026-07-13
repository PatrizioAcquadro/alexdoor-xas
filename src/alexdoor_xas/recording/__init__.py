"""Logging role: record every trial to the episode schema (docs/architecture.md)."""

from __future__ import annotations

from .episode import EpisodeBuffer, EpisodeMeta, EpisodeOutcome, EpisodeStep
from .writer import episode_filename, read_episode, write_episode

__all__ = [
    "EpisodeBuffer",
    "EpisodeMeta",
    "EpisodeOutcome",
    "EpisodeStep",
    "episode_filename",
    "read_episode",
    "write_episode",
]
