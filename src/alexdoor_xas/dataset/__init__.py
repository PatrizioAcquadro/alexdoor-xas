"""Dataset/model interface (Phase 3.0): load, validate, split, sample, normalize.

The frozen contract that lets learned baselines (ACT, Diffusion Policy, later
VLA conversion) consume Phase 2 episode datasets without special-case hacks.
See docs/dataset_interface.md. Pure numpy — no Isaac imports, h5py lazy.
"""

from __future__ import annotations

from .loader import (
    CONTACT_FLAG_KEY,
    DEFAULT_OBS_PRESET,
    OBS_PRESETS,
    A4ChunkDataset,
    A4EpisodeRecord,
    EpisodeDataset,
    EpisodeRecord,
    discover_episodes,
    expected_action_space,
    load_episode_record,
    obs_dim,
    obs_matrix,
    open_dataset,
    read_dataset_meta,
)
from .normalize import (
    NORM_STATS_FILENAME,
    STD_FLOOR,
    DatasetNormStats,
    NormStats,
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path,
    save_norm_stats,
)
from .sampling import (
    A4_FEATURE_DIM,
    A4_PHASE_VOCAB,
    BatchIterator,
    ChunkSample,
    ChunkSampler,
    chunk_features,
    collate_torch,
    episode_chunk_features,
)
from .splits import (
    DEFAULT_FRACTIONS,
    SPLIT_NAMES,
    load_splits,
    make_splits,
    save_splits,
    splits_path,
)
from .validate import (
    KNOWN_SCHEMA_VERSIONS,
    ValidationResult,
    validate_a4_dataset,
    validate_dataset,
    validate_dataset_dir,
    validate_episode,
)

__all__ = [
    "A4_FEATURE_DIM",
    "A4_PHASE_VOCAB",
    "CONTACT_FLAG_KEY",
    "DEFAULT_FRACTIONS",
    "DEFAULT_OBS_PRESET",
    "KNOWN_SCHEMA_VERSIONS",
    "NORM_STATS_FILENAME",
    "OBS_PRESETS",
    "SPLIT_NAMES",
    "STD_FLOOR",
    "A4ChunkDataset",
    "A4EpisodeRecord",
    "BatchIterator",
    "ChunkSample",
    "ChunkSampler",
    "DatasetNormStats",
    "EpisodeDataset",
    "EpisodeRecord",
    "NormStats",
    "ValidationResult",
    "chunk_features",
    "collate_torch",
    "compute_norm_stats",
    "discover_episodes",
    "episode_chunk_features",
    "expected_action_space",
    "load_episode_record",
    "load_norm_stats",
    "load_splits",
    "make_splits",
    "norm_stats_path",
    "obs_dim",
    "obs_matrix",
    "open_dataset",
    "read_dataset_meta",
    "save_norm_stats",
    "save_splits",
    "splits_path",
    "validate_a4_dataset",
    "validate_dataset",
    "validate_dataset_dir",
    "validate_episode",
]
