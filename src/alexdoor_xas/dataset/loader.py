"""Load A1-A4 exports into model-facing numpy records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import (
    ALL_ACTION_SPACES,
    ObjectCentricChunk,
)
from alexdoor_xas.recording import EpisodeBuffer, read_episode

_OBS_TABLES = ("proprio", "object_state", "contact")

_CONTACT_FLAG_KEY = "contact_flag"
_DOOR_YAW_SIN_KEY = "door_yaw_sin"
_DOOR_YAW_COS_KEY = "door_yaw_cos"

OBS_PRESETS: dict[str, tuple[str, ...]] = {
    "core": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
    ),
    "core_contact": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        _CONTACT_FLAG_KEY,
    ),
    # Core plus door origin relative to the robot base and wrap-free door yaw.
    "core_door_pose": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        "door_rel_pos_x",
        "door_rel_pos_y",
        "door_rel_pos_z",
        _DOOR_YAW_SIN_KEY,
        _DOOR_YAW_COS_KEY,
    ),
}
DEFAULT_OBS_PRESET = "core"


@dataclass(frozen=True)
class EpisodeRecord:
    """One loaded episode: stacked arrays + meta/outcome (model-facing view)."""

    episode_id: str
    action_space: str
    schema_version: str
    meta: dict[str, Any]
    t: np.ndarray  # (N,) seconds from episode start
    actions: np.ndarray  # (N, D) in `action_space`
    obs: dict[str, np.ndarray]  # per-step arrays, float64 (bools become 0/1)
    success: bool
    final_door_angle: float
    termination_reason: str
    environment_terminated: bool | None
    environment_truncated: bool | None
    extras: dict[str, Any]
    buffer: EpisodeBuffer = field(repr=False)  # source buffer (sanity checks)

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])


def _discover_episodes(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    return sorted(dataset_dir.glob("episode_*.hdf5"))


def _read_dataset_meta(dataset_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(dataset_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"dataset meta.json missing: {meta_path}")
    return json.loads(meta_path.read_text())


def _load_episode_record(path: str | Path) -> EpisodeRecord:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as h5:
        schema_version = str(h5.attrs.get("schema_version", ""))
    buffer = read_episode(path)
    if buffer.outcome is None:
        raise ValueError(f"episode {path} has no outcome")

    obs: dict[str, np.ndarray] = {}
    if buffer.steps:
        for table in _OBS_TABLES:
            for key, first in getattr(buffer.steps[0], table).items():
                if isinstance(first, str):
                    continue
                values = [getattr(step, table)[key] for step in buffer.steps]
                obs[key] = np.asarray(values, dtype=np.float64)

    return EpisodeRecord(
        episode_id=buffer.meta.episode_id,
        action_space=buffer.meta.action_space,
        schema_version=schema_version,
        meta=buffer.meta.to_dict(),
        t=np.array([step.t for step in buffer.steps], dtype=np.float64),
        actions=buffer.stacked(lambda s: s.action) if buffer.steps else np.zeros((0, 0)),
        obs=obs,
        success=buffer.outcome.success,
        final_door_angle=buffer.outcome.final_door_angle,
        termination_reason=buffer.outcome.termination_reason,
        environment_terminated=buffer.outcome.environment_terminated,
        environment_truncated=buffer.outcome.environment_truncated,
        extras=buffer.extras,
        buffer=buffer,
    )


def obs_matrix(record: EpisodeRecord, preset: str = DEFAULT_OBS_PRESET) -> np.ndarray:
    """Concatenate a frozen observation preset into one ``(N, obs_dim)`` matrix."""
    keys = _preset_keys(preset)
    columns: list[np.ndarray] = []
    for key in keys:
        array = _obs_key(record, key)
        if array is None:
            raise ValueError(
                f"episode {record.episode_id[:8]} (schema {record.schema_version!r}) "
                f"has no observation {key!r} required by preset {preset!r}"
            )
        columns.append(array.reshape(record.n_steps, -1))
    return np.concatenate(columns, axis=1) if columns else np.zeros((record.n_steps, 0))


def _preset_keys(preset: str) -> tuple[str, ...]:
    if preset not in OBS_PRESETS:
        raise ValueError(f"unknown obs preset {preset!r} (known: {sorted(OBS_PRESETS)})")
    return OBS_PRESETS[preset]


def _obs_key(record: EpisodeRecord, key: str) -> np.ndarray | None:
    if key == _CONTACT_FLAG_KEY:
        return record.obs.get("sensed", record.obs.get("inferred"))
    if key in (_DOOR_YAW_SIN_KEY, _DOOR_YAW_COS_KEY):
        yaw = record.obs.get("door_yaw_rad")
        if yaw is None:
            return None
        return np.sin(yaw) if key == _DOOR_YAW_SIN_KEY else np.cos(yaw)
    return record.obs.get(key)


class EpisodeDataset:
    """All episodes of one ``datasets/<task>/<action_space>/<version>/`` dir."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.meta = _read_dataset_meta(self.dataset_dir)
        paths = _discover_episodes(self.dataset_dir)
        if not paths:
            raise FileNotFoundError(f"no episode_*.hdf5 files in {self.dataset_dir}")
        self.records: list[EpisodeRecord] = [_load_episode_record(p) for p in paths]

    @property
    def action_space(self) -> str:
        return str(self.meta["action_space"])

    @property
    def task(self) -> str:
        return str(self.meta["task"])

    @property
    def action_dim(self) -> int:
        return self.records[0].action_dim

    @property
    def episode_ids(self) -> list[str]:
        return [record.episode_id for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EpisodeRecord:
        return self.records[index]

    def by_id(self, episode_id: str) -> EpisodeRecord:
        for record in self.records:
            if record.episode_id == episode_id:
                return record
        raise KeyError(f"episode {episode_id} not in {self.dataset_dir}")


@dataclass(frozen=True)
class A4EpisodeRecord:
    """One A4 episode: the ordered object-centric chunk log + meta/outcome."""

    episode_id: str
    action_space: str
    meta: dict[str, Any]
    chunks: tuple[ObjectCentricChunk, ...]
    success: bool
    final_door_angle: float
    termination_reason: str
    environment_terminated: bool | None
    environment_truncated: bool | None
    n_steps: int
    control_dt: float


class A4ChunkDataset:
    """All episodes of one A4 dataset dir (``episodes.jsonl`` + ``meta.json``)."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.meta = _read_dataset_meta(self.dataset_dir)
        jsonl = self.dataset_dir / "episodes.jsonl"
        if not jsonl.is_file():
            raise FileNotFoundError(f"A4 dataset episodes.jsonl missing: {jsonl}")
        self.records: list[A4EpisodeRecord] = []
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            self.records.append(_a4_record(json.loads(line)))
        if not self.records:
            raise ValueError(f"A4 dataset has no episodes: {jsonl}")

    @property
    def action_space(self) -> str:
        return str(self.meta["action_space"])

    @property
    def task(self) -> str:
        return str(self.meta["task"])

    @property
    def episode_ids(self) -> list[str]:
        return [record.episode_id for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> A4EpisodeRecord:
        return self.records[index]

    def by_id(self, episode_id: str) -> A4EpisodeRecord:
        for record in self.records:
            if record.episode_id == episode_id:
                return record
        raise KeyError(f"episode {episode_id} not in {self.dataset_dir}")


def _a4_record(data: dict[str, Any]) -> A4EpisodeRecord:
    if not isinstance(data, dict):
        raise ValueError("A4 episode record must be a JSON object")
    meta = _required_mapping(data, "meta", "A4 episode")
    episode_id = meta.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("A4 episode: meta.episode_id must be a non-empty string")
    episode_label = f"A4 episode {episode_id[:8]}"
    outcome = _required_mapping(data, "outcome", episode_label)
    _require_keys(
        meta,
        ("episode_id", "action_space", "seed", "robot", "scene", "policy", "control_dt"),
        episode_label,
    )
    _require_keys(
        outcome,
        ("success", "final_door_angle", "n_steps"),
        episode_label,
    )
    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"{episode_label}: chunks must be a list")
    return A4EpisodeRecord(
        episode_id=episode_id,
        action_space=str(meta["action_space"]),
        meta=meta,
        chunks=tuple(ObjectCentricChunk.from_dict(c) for c in chunks),
        success=_required_bool(outcome["success"], f"{episode_label}: outcome.success"),
        final_door_angle=float(outcome["final_door_angle"]),
        termination_reason=str(outcome.get("termination_reason", "not_recorded")),
        environment_terminated=_optional_bool(outcome.get("environment_terminated")),
        environment_truncated=_optional_bool(outcome.get("environment_truncated")),
        n_steps=int(outcome["n_steps"]),
        control_dt=float(meta["control_dt"]),
    )


def _required_mapping(data: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: missing required {key} object")
    return dict(value)


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{label}: missing required keys {missing}")


def _required_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("environment termination flags must be JSON booleans or null")
    return value


def _expected_action_space(dataset_dir: str | Path) -> str | None:
    parent = Path(dataset_dir).resolve().parent.name
    return parent if parent in ALL_ACTION_SPACES else None
