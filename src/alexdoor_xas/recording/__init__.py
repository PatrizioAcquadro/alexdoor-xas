"""Record every trial to the schema in the episode-and-dataset wiki contract."""

from __future__ import annotations

from .episode import (
    LEGACY_TERMINATION_REASON,
    TERMINATION_REASONS,
    EpisodeBuffer,
    EpisodeMeta,
    EpisodeOutcome,
    EpisodeStep,
)
from .writer import (
    LEGACY_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    episode_filename,
    read_episode,
    write_episode,
)

__all__ = [
    "LEGACY_SCHEMA_VERSIONS",
    "LEGACY_TERMINATION_REASON",
    "SCHEMA_VERSION",
    "TERMINATION_REASONS",
    "EpisodeBuffer",
    "EpisodeMeta",
    "EpisodeOutcome",
    "EpisodeStep",
    "episode_filename",
    "read_episode",
    "write_episode",
]
