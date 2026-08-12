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
from alexdoor_xas.eval.sanity import check_alex_episode
from alexdoor_xas.recording import LEGACY_TERMINATION_REASON, TERMINATION_REASONS

from .loader import (
    A4ChunkDataset,
    A4EpisodeRecord,
    EpisodeDataset,
    EpisodeRecord,
    expected_action_space,
    obs_matrix,
)
from .sampling import A4_FEATURE_DIM, A4_PHASE_VOCAB, episode_chunk_features

KNOWN_SCHEMA_VERSIONS = ("phase2.v0", "phase2.v1", "phase2.v2")
REQUIRED_DATASET_META_KEYS = (
    "task",
    "action_space",
    "n_episodes",
    "seeds",
    "robot",
    "scene",
    "policy",
)
REQUIRED_EPISODE_META_KEYS = (
    "episode_id",
    "task",
    "action_space",
    "robot",
    "scene",
    "policy",
    "seed",
    "control_dt",
)
EXPECTED_CONTACT_SOURCES = ("inferred_geometric", "force_sensor+geometric")
TIMESTAMP_ATOL_S = 1e-7


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
    _check_required_keys(record.meta, REQUIRED_EPISODE_META_KEYS, label, result)

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

    actions = np.asarray(record.actions)
    if actions.ndim != 2:
        result.errors.append(f"{label}: action tensor must have rank 2 (N, D), got {actions.shape}")
    n_steps = int(actions.shape[0]) if actions.ndim >= 1 else 0

    control_dt = _positive_finite_control_dt(record.meta, label, result)
    if not np.isfinite(record.final_door_angle):
        result.errors.append(f"{label}: final_door_angle must be finite")

    if n_steps == 0:
        result.errors.append(f"{label}: episode has no recorded steps")
        return result

    if actions.ndim == 2:
        _check_action_dim(record, result, label)
        if record.action_space == A3_OBJ_REL_EE_DELTA:
            _check_a3_actions(record, result, label)

    outcome_steps = record.buffer.outcome.n_steps if record.buffer.outcome else -1
    if not (n_steps == len(record.t) == outcome_steps):
        result.errors.append(
            f"{label}: inconsistent step counts (actions {n_steps}, "
            f"t {len(record.t)}, outcome.n_steps {outcome_steps})"
        )
    for key, array in record.obs.items():
        array = np.asarray(array)
        if array.ndim == 0:
            result.errors.append(
                f"{label}: obs {key!r} must have a leading step dimension, got shape {array.shape}"
            )
            continue
        if array.shape[0] != n_steps:
            result.errors.append(
                f"{label}: obs {key!r} has {array.shape[0]} steps, expected {n_steps}"
            )

    if not np.isfinite(actions).all():
        result.errors.append(f"{label}: non-finite action values")
    for key, array in record.obs.items():
        if not np.isfinite(array).all():
            result.errors.append(f"{label}: non-finite obs {key!r} values")
    _check_timestamps(record.t, control_dt, result, label)
    _check_contact_semantics(record, result, label)
    _check_obs_ref_consistency(record, result, label)

    try:
        obs_matrix(record, "core")
    except ValueError as exc:
        result.errors.append(f"{label}: core obs preset failed: {exc}")

    _check_termination_data(record, result, label, legacy=record.schema_version != "phase2.v2")

    # Alex V2 force-sensing episodes get the full rollout sanity checks.
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
    _check_required_keys(dataset.meta, REQUIRED_DATASET_META_KEYS, "meta.json", result)
    dataset_space = dataset.meta.get("action_space")

    if expected_space is not None and dataset_space != expected_space:
        result.errors.append(
            f"meta.json action_space {dataset_space!r} does not match the "
            f"directory tag {expected_space!r}"
        )
    try:
        n_declared = int(dataset.meta.get("n_episodes", -1))
    except (TypeError, ValueError):
        n_declared = -1
        result.errors.append("meta.json n_episodes must be an integer")
    if n_declared != len(dataset):
        result.errors.append(
            f"meta.json declares {n_declared} episodes, found {len(dataset)}"
        )
    ids = dataset.episode_ids
    if len(set(ids)) != len(ids):
        result.errors.append("duplicate episode ids in dataset")

    dims = {record.action_dim for record in dataset.records if np.asarray(record.actions).ndim == 2}
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
    _check_required_keys(dataset.meta, REQUIRED_DATASET_META_KEYS, "meta.json", result)
    dataset_space = dataset.meta.get("action_space")
    if expected_space is not None and dataset_space != expected_space:
        result.errors.append(
            f"meta.json action_space {dataset_space!r} does not match the "
            f"directory tag {expected_space!r}"
        )
    try:
        n_declared = int(dataset.meta.get("n_episodes", -1))
    except (TypeError, ValueError):
        n_declared = -1
        result.errors.append("meta.json n_episodes must be an integer")
    if n_declared != len(dataset):
        result.errors.append(
            f"meta.json declares {n_declared} episodes, found {len(dataset)}"
        )
    ids = dataset.episode_ids
    duplicates = sorted({episode_id for episode_id in ids if ids.count(episode_id) > 1})
    if duplicates:
        result.errors.append(f"duplicate A4 episode ids: {duplicates}")

    for record in dataset.records:
        label = f"episode {record.episode_id[:8]}"
        if record.action_space != dataset.action_space:
            result.errors.append(
                f"{label}: action_space {record.action_space!r} != dataset "
                f"{dataset.action_space!r}"
            )
        _check_a4_outcome(record, result, label)
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
            if len(chunk.contact_target_panel) != 3:
                result.errors.append(
                    f"{label}: chunk {i} contact_target_panel must have length 3, "
                    f"got {len(chunk.contact_target_panel)}"
                )
            values = (*chunk.contact_target_panel, chunk.motion_hinge_delta_rad)
            if not np.isfinite(values).all():
                result.errors.append(f"{label}: chunk {i} has non-finite values")
        try:
            features = episode_chunk_features(record)
        except ValueError as exc:
            result.errors.append(f"{label}: A4 feature encoding failed: {exc}")
        else:
            if features.ndim != 2 or features.shape[1] != A4_FEATURE_DIM:
                result.errors.append(
                    f"{label}: A4 feature encoding shape {features.shape} "
                    f"!= (chunks, {A4_FEATURE_DIM})"
                )
            if not np.isfinite(features).all():
                result.errors.append(f"{label}: A4 feature encoding has non-finite values")
        duration_ticks = sum(chunk.duration_ticks for chunk in record.chunks)
        if record.success and duration_ticks != record.n_steps:
            result.warnings.append(
                f"{label}: successful A4 chunk durations sum to {duration_ticks}, "
                f"outcome.n_steps is {record.n_steps}"
            )
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
    try:
        if isinstance(dataset, A4ChunkDataset):
            return validate_a4_dataset(dataset)
        return validate_dataset(dataset)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result = ValidationResult()
        result.errors.append(f"{dataset_dir}: failed to validate dataset: {exc}")
        return result


def validate_matched_action_space_datasets(
    hdf5_datasets: dict[str, EpisodeDataset],
    a4_dataset: A4ChunkDataset | None = None,
) -> ValidationResult:
    """Validate same-ID episodes are matched in content, not only set membership."""
    result = ValidationResult()
    if not hdf5_datasets:
        result.errors.append("no HDF5 datasets to compare")
        return result

    reference_space = next(iter(hdf5_datasets))
    reference = hdf5_datasets[reference_space]
    reference_ids = set(reference.episode_ids)
    for space, dataset in hdf5_datasets.items():
        ids = set(dataset.episode_ids)
        if ids != reference_ids:
            result.errors.append(f"episode ids of {space} differ from {reference_space}")
            continue
        for episode_id in sorted(reference_ids):
            _compare_hdf5_records(
                reference.by_id(episode_id),
                dataset.by_id(episode_id),
                reference_space,
                space,
                result,
            )
    if a4_dataset is not None:
        ids = set(a4_dataset.episode_ids)
        if ids != reference_ids:
            result.errors.append(f"episode ids of A4 differ from {reference_space}")
        else:
            for episode_id in sorted(reference_ids):
                _compare_a4_record(
                    reference.by_id(episode_id),
                    a4_dataset.by_id(episode_id),
                    reference_space,
                    result,
                )
    return result


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


def _check_required_keys(
    data: dict[str, object], keys: tuple[str, ...], label: str, result: ValidationResult
) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        result.errors.append(f"{label}: missing required keys {missing}")


def _positive_finite_control_dt(
    meta: dict[str, object], label: str, result: ValidationResult
) -> float | None:
    if "control_dt" not in meta:
        return None
    try:
        control_dt = float(meta["control_dt"])
    except (TypeError, ValueError):
        result.errors.append(f"{label}: control_dt must be a positive finite number")
        return None
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        result.errors.append(f"{label}: control_dt must be positive and finite, got {control_dt!r}")
        return None
    return control_dt


def _check_timestamps(
    t: np.ndarray, control_dt: float | None, result: ValidationResult, label: str
) -> None:
    t = np.asarray(t, dtype=np.float64)
    if not np.isfinite(t).all():
        result.errors.append(f"{label}: non-finite timestamps")
        return
    if t.size == 0:
        return
    if abs(float(t[0])) > TIMESTAMP_ATOL_S:
        result.errors.append(f"{label}: timestamps must start at 0.0, got {t[0]!r}")
    if t.size > 1 and not (np.diff(t) > 0).all():
        result.errors.append(f"{label}: step times are not strictly increasing")
    if control_dt is not None and t.size > 1:
        diffs = np.diff(t)
        if not np.allclose(diffs, control_dt, rtol=1e-6, atol=TIMESTAMP_ATOL_S):
            result.errors.append(
                f"{label}: timestamp deltas do not match meta.control_dt={control_dt}"
            )


def _check_contact_semantics(
    record: EpisodeRecord, result: ValidationResult, label: str
) -> None:
    for key in ("inferred", "sensed"):
        if key in record.obs:
            values = np.asarray(record.obs[key])
            if not np.isin(values, (0.0, 1.0)).all():
                result.errors.append(f"{label}: contact flag {key!r} must be binary")
    for i, step in enumerate(record.buffer.steps):
        source = step.contact.get("source")
        if source not in EXPECTED_CONTACT_SOURCES:
            result.errors.append(
                f"{label}: contact source at step {i} must be one of "
                f"{EXPECTED_CONTACT_SOURCES}, got {source!r}"
            )
            break
        if source == "force_sensor+geometric":
            if "sensed" not in step.contact or "force_n" not in step.contact:
                result.errors.append(
                    f"{label}: force contact source at step {i} requires sensed and force_n"
                )
                break
            if not isinstance(step.contact["sensed"], bool):
                result.errors.append(f"{label}: contact.sensed at step {i} must be boolean")
                break
            try:
                force = float(step.contact["force_n"])
            except (TypeError, ValueError):
                result.errors.append(f"{label}: contact.force_n at step {i} must be numeric")
                break
            if not np.isfinite(force) or force < 0.0:
                result.errors.append(f"{label}: contact.force_n at step {i} must be finite >= 0")
                break
        elif "sensed" in step.contact or "force_n" in step.contact:
            result.errors.append(
                f"{label}: inferred-only contact source at step {i} must not carry "
                "force-sensor fields"
            )
            break


def _check_obs_ref_consistency(
    record: EpisodeRecord, result: ValidationResult, label: str
) -> None:
    if not record.buffer.steps:
        return
    comparisons = (
        ("door_angle_rad", "object_state", "door_angle_rad"),
        ("door_angular_velocity_rad_s", "object_state", "door_angular_velocity_rad_s"),
        ("ee_pos_x_m", "proprio", "ee_pos_w", 0),
        ("ee_pos_y_m", "proprio", "ee_pos_w", 1),
        ("ee_pos_z_m", "proprio", "ee_pos_w", 2),
    )
    for spec in comparisons:
        obs_ref_key, table, key, *index = spec
        values = []
        refs = []
        for step in record.buffer.steps:
            if obs_ref_key not in step.obs_ref or key not in getattr(step, table):
                continue
            value = getattr(step, table)[key]
            if index:
                value = np.asarray(value, dtype=np.float64)[index[0]]
            values.append(float(value))
            refs.append(float(step.obs_ref[obs_ref_key]))
        if values and not np.allclose(values, refs, rtol=1e-6, atol=1e-9):
            result.errors.append(f"{label}: obs_ref {obs_ref_key!r} disagrees with {table}.{key}")


def _check_a3_actions(record: EpisodeRecord, result: ValidationResult, label: str) -> None:
    expected = record.extras.get("action_door_frame")
    if expected is None:
        result.warnings.append(f"{label}: A3 episode has no action_door_frame extra")
        return
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != record.actions.shape:
        result.errors.append(
            f"{label}: A3 actions shape {record.actions.shape} does not match "
            f"extras['action_door_frame'] shape {expected.shape}"
        )
        return
    if not np.allclose(record.actions, expected, rtol=1e-6, atol=1e-9):
        result.errors.append(f"{label}: A3 actions do not match extras['action_door_frame']")
    frame_pos = np.asarray(record.extras.get("door_frame_pos_w", []), dtype=np.float64)
    frame_quat = np.asarray(record.extras.get("door_frame_quat_w_xyzw", []), dtype=np.float64)
    if frame_pos.shape != (3,) or not np.isfinite(frame_pos).all():
        result.errors.append(f"{label}: A3 door_frame_pos_w extra must be finite shape (3,)")
    if frame_quat.shape != (4,) or not np.isfinite(frame_quat).all():
        result.errors.append(f"{label}: A3 door_frame_quat_w_xyzw extra must be finite shape (4,)")
    elif not np.isclose(np.linalg.norm(frame_quat), 1.0, rtol=1e-5, atol=1e-5):
        result.errors.append(f"{label}: A3 door_frame_quat_w_xyzw must be normalized")


def _check_a4_outcome(
    record: A4EpisodeRecord, result: ValidationResult, label: str
) -> None:
    _positive_finite_control_dt(record.meta, label, result)
    if not np.isfinite(record.final_door_angle):
        result.errors.append(f"{label}: final_door_angle must be finite")
    if record.n_steps <= 0:
        result.errors.append(f"{label}: outcome.n_steps must be positive")
    _check_termination_data(
        record, result, label, legacy=record.termination_reason == "not_recorded"
    )


def _check_termination_data(record, result: ValidationResult, label: str, *, legacy: bool) -> None:
    allowed = (*TERMINATION_REASONS, LEGACY_TERMINATION_REASON)
    if record.termination_reason not in allowed:
        result.errors.append(
            f"{label}: unknown termination_reason {record.termination_reason!r}"
        )
    if legacy:
        if record.termination_reason != LEGACY_TERMINATION_REASON:
            result.errors.append(f"{label}: legacy episode termination_reason must be not_recorded")
        if record.environment_terminated is not None or record.environment_truncated is not None:
            result.errors.append(f"{label}: legacy environment termination flags must be unknown")
        return
    if not isinstance(record.environment_terminated, bool) or not isinstance(
        record.environment_truncated, bool
    ):
        result.errors.append(f"{label}: phase2.v2 environment termination flags must be booleans")
        return
    if record.termination_reason == "environment_terminated" and not record.environment_terminated:
        result.errors.append(f"{label}: environment_terminated reason requires its factual flag")
    if record.termination_reason == "environment_truncated" and not record.environment_truncated:
        result.errors.append(f"{label}: environment_truncated reason requires its factual flag")


def _compare_hdf5_records(
    reference: EpisodeRecord,
    candidate: EpisodeRecord,
    reference_space: str,
    candidate_space: str,
    result: ValidationResult,
) -> None:
    if reference is candidate or reference_space == candidate_space:
        return
    label = f"episode {reference.episode_id[:8]} {candidate_space} vs {reference_space}"
    for key in ("seed", "robot", "scene", "policy"):
        if reference.meta.get(key) != candidate.meta.get(key):
            result.errors.append(
                f"{label}: meta.{key} differs "
                f"({candidate.meta.get(key)!r} != {reference.meta.get(key)!r})"
            )
    if reference.n_steps != candidate.n_steps:
        result.errors.append(
            f"{label}: n_steps differs ({candidate.n_steps} != {reference.n_steps})"
        )
        return
    if not np.allclose(candidate.t, reference.t, rtol=1e-6, atol=TIMESTAMP_ATOL_S):
        result.errors.append(f"{label}: timestamps differ")
    if reference.success != candidate.success:
        result.errors.append(f"{label}: outcome.success differs")
    if not np.isclose(candidate.final_door_angle, reference.final_door_angle, rtol=1e-6, atol=1e-9):
        result.errors.append(f"{label}: outcome.final_door_angle differs")
    if (
        reference.termination_reason,
        reference.environment_terminated,
        reference.environment_truncated,
    ) != (
        candidate.termination_reason,
        candidate.environment_terminated,
        candidate.environment_truncated,
    ):
        result.errors.append(f"{label}: factual termination data differs")
    try:
        reference_obs = obs_matrix(reference, "core")
        candidate_obs = obs_matrix(candidate, "core")
    except ValueError as exc:
        result.errors.append(f"{label}: core obs comparison failed: {exc}")
        return
    if not np.allclose(candidate_obs, reference_obs, rtol=1e-6, atol=1e-9):
        result.errors.append(f"{label}: core low-dim observations differ")


def _compare_a4_record(
    reference: EpisodeRecord,
    candidate: A4EpisodeRecord,
    reference_space: str,
    result: ValidationResult,
) -> None:
    label = f"episode {reference.episode_id[:8]} A4 vs {reference_space}"
    for key in ("seed", "robot", "scene", "policy"):
        if reference.meta.get(key) != candidate.meta.get(key):
            result.errors.append(
                f"{label}: meta.{key} differs "
                f"({candidate.meta.get(key)!r} != {reference.meta.get(key)!r})"
            )
    if reference.n_steps != candidate.n_steps:
        result.errors.append(
            f"{label}: n_steps differs ({candidate.n_steps} != {reference.n_steps})"
        )
    if reference.success != candidate.success:
        result.errors.append(f"{label}: outcome.success differs")
    if not np.isclose(candidate.final_door_angle, reference.final_door_angle, rtol=1e-6, atol=1e-9):
        result.errors.append(f"{label}: outcome.final_door_angle differs")
    if (
        reference.termination_reason,
        reference.environment_terminated,
        reference.environment_truncated,
    ) != (
        candidate.termination_reason,
        candidate.environment_terminated,
        candidate.environment_truncated,
    ):
        result.errors.append(f"{label}: factual termination data differs")


__all__ = [
    "KNOWN_SCHEMA_VERSIONS",
    "ValidationResult",
    "validate_a4_dataset",
    "validate_dataset",
    "validate_dataset_dir",
    "validate_episode",
    "validate_matched_action_space_datasets",
]
