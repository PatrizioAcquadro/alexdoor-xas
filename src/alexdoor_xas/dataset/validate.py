"""Episode/dataset validation against the frozen Phase 2 schema (Phase 3.0).

Confirms exported datasets are consumable by learned baselines *before* any
training code exists: known schema version, action-space/directory agreement,
action dimensionality, step-count and timestamp consistency, finite values,
and the frozen ``core`` observation preset. Force-sensing (Alex) episodes are
additionally passed through the existing rollout sanity checks
(:func:`alexdoor_xas.eval.sanity.check_alex_episode`) — deep joint/contact
validation is not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    EE_DELTA_DIM,
)
from alexdoor_xas.eval.failures import FAILURE_LABELS
from alexdoor_xas.eval.sanity import check_alex_episode

from .loader import (
    A4ChunkDataset,
    EpisodeDataset,
    EpisodeRecord,
    expected_action_space,
    obs_matrix,
)
from .sampling import A4_PHASE_VOCAB

KNOWN_SCHEMA_VERSIONS = ("phase2.v0", "phase2.v1")


@dataclass
class ValidationResult:
    """Hard failures + soft warnings (same shape as ``eval.sanity.SanityResult``)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def validate_episode(
    record: EpisodeRecord, expected_space: str | None = None
) -> ValidationResult:
    """Validate one loaded episode against the frozen schema contract."""
    result = ValidationResult()
    label = f"episode {record.episode_id[:8]}"

    if record.schema_version not in KNOWN_SCHEMA_VERSIONS:
        result.errors.append(
            f"{label}: unknown schema_version {record.schema_version!r} "
            f"(known: {KNOWN_SCHEMA_VERSIONS})"
        )
    if expected_space is not None and record.action_space != expected_space:
        result.errors.append(
            f"{label}: meta.action_space {record.action_space!r} does not match "
            f"the dataset directory tag {expected_space!r}"
        )

    if record.n_steps == 0:
        result.errors.append(f"{label}: episode has no recorded steps")
        return result

    _check_action_dim(record, result, label)

    outcome_steps = record.buffer.outcome.n_steps if record.buffer.outcome else -1
    if not (record.n_steps == len(record.t) == outcome_steps):
        result.errors.append(
            f"{label}: inconsistent step counts (actions {record.n_steps}, "
            f"t {len(record.t)}, outcome.n_steps {outcome_steps})"
        )
    for key, array in record.obs.items():
        if array.shape[0] != record.n_steps:
            result.errors.append(
                f"{label}: obs {key!r} has {array.shape[0]} steps, expected {record.n_steps}"
            )

    if not np.isfinite(record.actions).all():
        result.errors.append(f"{label}: non-finite action values")
    for key, array in record.obs.items():
        if not np.isfinite(array).all():
            result.errors.append(f"{label}: non-finite obs {key!r} values")
    if len(record.t) > 1 and not (np.diff(record.t) > 0).all():
        result.errors.append(f"{label}: step times are not strictly increasing")

    try:
        obs_matrix(record, "core")
    except ValueError as exc:
        result.errors.append(f"{label}: core obs preset failed: {exc}")

    if record.success and record.failure_label is not None:
        result.warnings.append(
            f"{label}: success episode carries failure_label {record.failure_label!r}"
        )
    if record.failure_label is not None and record.failure_label not in FAILURE_LABELS:
        result.warnings.append(
            f"{label}: failure_label {record.failure_label!r} not in the frozen vocabulary"
        )

    # Force-sensing episodes get the full Phase 2.5 rollout sanity checks.
    if "joint_pos" in record.obs:
        sanity = check_alex_episode(record.buffer)
        result.errors.extend(sanity.errors)
        result.warnings.extend(sanity.warnings)

    return result


def validate_dataset(
    dataset: EpisodeDataset, expected_space: str | None = None
) -> ValidationResult:
    """Validate every episode plus dataset-level consistency."""
    result = ValidationResult()
    expected_space = expected_space or expected_action_space(dataset.dataset_dir)

    if expected_space is not None and dataset.action_space != expected_space:
        result.errors.append(
            f"meta.json action_space {dataset.action_space!r} does not match the "
            f"directory tag {expected_space!r}"
        )
    n_declared = int(dataset.meta.get("n_episodes", -1))
    if n_declared != len(dataset):
        result.errors.append(
            f"meta.json declares {n_declared} episodes, found {len(dataset)}"
        )
    ids = dataset.episode_ids
    if len(set(ids)) != len(ids):
        result.errors.append("duplicate episode ids in dataset")

    dims = {record.action_dim for record in dataset.records}
    if len(dims) > 1:
        result.errors.append(f"episodes disagree on action dim: {sorted(dims)}")

    for record in dataset.records:
        result.merge(validate_episode(record, expected_space))
    return result


def validate_a4_dataset(
    dataset: A4ChunkDataset, expected_space: str | None = None
) -> ValidationResult:
    """Validate an A4 chunk dataset (chunks already parsed at load time)."""
    result = ValidationResult()
    expected_space = expected_space or expected_action_space(dataset.dataset_dir)
    if expected_space is not None and dataset.action_space != expected_space:
        result.errors.append(
            f"meta.json action_space {dataset.action_space!r} does not match the "
            f"directory tag {expected_space!r}"
        )
    n_declared = int(dataset.meta.get("n_episodes", -1))
    if n_declared != len(dataset):
        result.errors.append(
            f"meta.json declares {n_declared} episodes, found {len(dataset)}"
        )

    for record in dataset.records:
        label = f"episode {record.episode_id[:8]}"
        if record.action_space != dataset.action_space:
            result.errors.append(
                f"{label}: action_space {record.action_space!r} != dataset "
                f"{dataset.action_space!r}"
            )
        if not record.chunks:
            result.errors.append(f"{label}: no A4 chunks recorded")
        for i, chunk in enumerate(record.chunks):
            if chunk.phase not in A4_PHASE_VOCAB:
                result.errors.append(
                    f"{label}: chunk {i} has unknown phase {chunk.phase!r}"
                )
            if chunk.duration_ticks <= 0:
                result.errors.append(
                    f"{label}: chunk {i} ({chunk.phase}) has duration_ticks "
                    f"{chunk.duration_ticks} <= 0"
                )
            values = (*chunk.contact_target_panel, chunk.motion_hinge_delta_rad)
            if not np.isfinite(values).all():
                result.errors.append(f"{label}: chunk {i} has non-finite values")
    return result


def validate_dataset_dir(dataset_dir: str | Path) -> ValidationResult:
    """Open + validate one dataset directory with the right loader."""
    from .loader import open_dataset

    try:
        dataset = open_dataset(dataset_dir)
    except (OSError, ValueError, KeyError) as exc:
        result = ValidationResult()
        result.errors.append(f"{dataset_dir}: failed to open dataset: {exc}")
        return result
    if isinstance(dataset, A4ChunkDataset):
        return validate_a4_dataset(dataset)
    return validate_dataset(dataset)


def _check_action_dim(
    record: EpisodeRecord, result: ValidationResult, label: str
) -> None:
    if record.action_space in (A2_EE_DELTA, A3_OBJ_REL_EE_DELTA):
        if record.action_dim != EE_DELTA_DIM:
            result.errors.append(
                f"{label}: {record.action_space} actions must be {EE_DELTA_DIM}-dim, "
                f"got {record.action_dim}"
            )
    elif record.action_space == A1_JOINT_DELTA:
        joint_names = record.extras.get("joint_names")
        if joint_names is None:
            result.warnings.append(
                f"{label}: A1 episode has no joint_names extra to check the action dim"
            )
        elif record.action_dim != len(joint_names):
            result.errors.append(
                f"{label}: A1 actions are {record.action_dim}-dim but the robot has "
                f"{len(joint_names)} joints"
            )


__all__ = [
    "KNOWN_SCHEMA_VERSIONS",
    "ValidationResult",
    "validate_a4_dataset",
    "validate_dataset",
    "validate_dataset_dir",
    "validate_episode",
]
