"""Validate dataset integrity and matched A1-A4 semantics."""

from __future__ import annotations

import numpy as np

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_PHASE_VOCAB,
    EE_DELTA_DIM,
)
from alexdoor_xas.eval.sanity import SanityResult, check_alex_episode
from alexdoor_xas.recording import (
    LEGACY_SCHEMA_VERSIONS,
    LEGACY_TERMINATION_REASON,
    SCHEMA_VERSION,
    TERMINATION_REASONS,
)

from .loader import (
    A4ChunkDataset,
    A4EpisodeRecord,
    EpisodeDataset,
    EpisodeRecord,
    _expected_action_space,
    obs_matrix,
)

_KNOWN_SCHEMA_VERSIONS = (*LEGACY_SCHEMA_VERSIONS, SCHEMA_VERSION)
REQUIRED_DATASET_META_KEYS = (
    "task",
    "action_space",
    "n_episodes",
    "seeds",
    "robot",
    "scene",
    "policy",
)
EXPECTED_CONTACT_SOURCES = ("inferred_geometric", "force_sensor+geometric")
TIMESTAMP_ATOL_S = 1e-7


def validate_episode(record: EpisodeRecord, expected_space: str | None = None) -> SanityResult:
    """Validate one loaded episode against the frozen schema contract."""
    result = SanityResult()
    label = f"episode {record.episode_id[:8]}"

    if record.schema_version not in _KNOWN_SCHEMA_VERSIONS:
        result.errors.append(
            f"{label}: unknown schema_version {record.schema_version!r} "
            f"(known: {_KNOWN_SCHEMA_VERSIONS})"
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
    try:
        obs_matrix(record, "core")
    except ValueError as exc:
        result.errors.append(f"{label}: core obs preset failed: {exc}")

    _check_termination_data(record, result, label, legacy=record.schema_version != SCHEMA_VERSION)

    # Alex V2 force-sensing episodes get the full rollout sanity checks.
    if "joint_pos" in record.obs:
        sanity = check_alex_episode(record.buffer)
        _merge(result, sanity)

    return result


def validate_dataset(dataset: EpisodeDataset, expected_space: str | None = None) -> SanityResult:
    """Validate every episode plus dataset-level consistency."""
    result = SanityResult()
    expected_space = expected_space or _expected_action_space(dataset.dataset_dir)
    _check_dataset_header(dataset, expected_space, result)

    dims = {record.action_dim for record in dataset.records if np.asarray(record.actions).ndim == 2}
    if len(dims) > 1:
        result.errors.append(f"episodes disagree on action dim: {sorted(dims)}")

    for record in dataset.records:
        _merge(result, validate_episode(record, expected_space))
    return result


def validate_a4_dataset(dataset: A4ChunkDataset, expected_space: str | None = None) -> SanityResult:
    """Validate an A4 chunk dataset (chunks already parsed at load time)."""
    result = SanityResult()
    expected_space = expected_space or _expected_action_space(dataset.dataset_dir)
    dataset_space = dataset.meta.get("action_space")
    _check_dataset_header(dataset, expected_space, result)

    for record in dataset.records:
        label = f"episode {record.episode_id[:8]}"
        if record.action_space != dataset_space:
            result.errors.append(
                f"{label}: action_space {record.action_space!r} != dataset {dataset_space!r}"
            )
        _check_a4_outcome(record, result, label)
        if not record.chunks:
            result.errors.append(f"{label}: no A4 chunks recorded")
        for i, chunk in enumerate(record.chunks):
            if chunk.phase not in A4_PHASE_VOCAB:
                result.errors.append(f"{label}: chunk {i} has unknown phase {chunk.phase!r}")
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
        duration_ticks = sum(chunk.duration_ticks for chunk in record.chunks)
        if record.success and duration_ticks != record.n_steps:
            result.warnings.append(
                f"{label}: successful A4 chunk durations sum to {duration_ticks}, "
                f"outcome.n_steps is {record.n_steps}"
            )
    return result


def validate_matched_action_space_datasets(
    hdf5_datasets: dict[str, EpisodeDataset],
    a4_dataset: A4ChunkDataset | None = None,
) -> SanityResult:
    """Validate same-ID episodes are matched in content, not only set membership."""
    result = SanityResult()
    if not hdf5_datasets:
        result.errors.append("no HDF5 datasets to compare")
        return result

    reference_space = next(iter(hdf5_datasets))
    reference = hdf5_datasets[reference_space]
    reference_ids = set(reference.episode_ids)
    reference_by_id = {record.episode_id: record for record in reference.records}
    for space, dataset in hdf5_datasets.items():
        ids = set(dataset.episode_ids)
        if ids != reference_ids:
            result.errors.append(f"episode ids of {space} differ from {reference_space}")
            continue
        candidate_by_id = {record.episode_id: record for record in dataset.records}
        for episode_id in sorted(reference_ids):
            _compare_hdf5_records(
                reference_by_id[episode_id],
                candidate_by_id[episode_id],
                reference_space,
                space,
                result,
            )
    if a4_dataset is not None:
        ids = set(a4_dataset.episode_ids)
        if ids != reference_ids:
            result.errors.append(f"episode ids of A4 differ from {reference_space}")
        else:
            a4_by_id = {record.episode_id: record for record in a4_dataset.records}
            for episode_id in sorted(reference_ids):
                _compare_a4_record(
                    reference_by_id[episode_id],
                    a4_by_id[episode_id],
                    reference_space,
                    result,
                )
    return result


def _merge(result: SanityResult, other: SanityResult) -> None:
    result.errors.extend(other.errors)
    result.warnings.extend(other.warnings)


def _check_dataset_header(
    dataset: EpisodeDataset | A4ChunkDataset,
    expected_space: str | None,
    result: SanityResult,
) -> None:
    _check_required_keys(dataset.meta, REQUIRED_DATASET_META_KEYS, "meta.json", result)
    dataset_space = dataset.meta.get("action_space")
    if expected_space is not None and dataset_space != expected_space:
        result.errors.append(
            f"meta.json action_space {dataset_space!r} does not match "
            f"directory tag {expected_space!r}"
        )
    try:
        n_declared = int(dataset.meta.get("n_episodes", -1))
    except (TypeError, ValueError):
        result.errors.append("meta.json n_episodes must be an integer")
    else:
        if n_declared != len(dataset):
            result.errors.append(f"meta.json declares {n_declared} episodes, found {len(dataset)}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for episode_id in dataset.episode_ids:
        if episode_id in seen:
            duplicates.add(episode_id)
        seen.add(episode_id)
    if duplicates:
        result.errors.append(f"duplicate episode ids: {sorted(duplicates)}")


def _check_action_dim(record: EpisodeRecord, result: SanityResult, label: str) -> None:
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
    data: dict[str, object], keys: tuple[str, ...], label: str, result: SanityResult
) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        result.errors.append(f"{label}: missing required keys {missing}")


def _positive_finite_control_dt(
    meta: dict[str, object], label: str, result: SanityResult
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
    t: np.ndarray, control_dt: float | None, result: SanityResult, label: str
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


def _check_contact_semantics(record: EpisodeRecord, result: SanityResult, label: str) -> None:
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


def _check_a3_actions(record: EpisodeRecord, result: SanityResult, label: str) -> None:
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


def _check_a4_outcome(record: A4EpisodeRecord, result: SanityResult, label: str) -> None:
    _positive_finite_control_dt(record.meta, label, result)
    if not np.isfinite(record.final_door_angle):
        result.errors.append(f"{label}: final_door_angle must be finite")
    if record.n_steps <= 0:
        result.errors.append(f"{label}: outcome.n_steps must be positive")
    _check_termination_data(
        record, result, label, legacy=record.termination_reason == "not_recorded"
    )


def _check_termination_data(record, result: SanityResult, label: str, *, legacy: bool) -> None:
    allowed = (*TERMINATION_REASONS, LEGACY_TERMINATION_REASON)
    if record.termination_reason not in allowed:
        result.errors.append(f"{label}: unknown termination_reason {record.termination_reason!r}")
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
    result: SanityResult,
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
    result: SanityResult,
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
