"""Read exported episode datasets into model-ready numpy records (Phase 3.0).

One dataset directory (``datasets/<task>/<action_space>/<version>/``) is loaded
into :class:`EpisodeDataset` (A1/A2/A3, HDF5 episodes) or :class:`A4ChunkDataset`
(A4, JSON lines). Episodes are exposed as :class:`EpisodeRecord`: per-episode
stacked arrays plus meta/outcome, so learned baselines never touch the storage
container directly. Observation vectors are built from **frozen named presets**
(:data:`OBS_PRESETS`) — the Phase 3.0 dataset/model interface freeze
(``knowledge/wiki/topics/episode-and-dataset-contracts.md``).

Pure numpy; h5py is imported lazily (same policy as ``recording/writer.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import (
    A4_OBJ_CENTRIC_CHUNK,
    ALL_ACTION_SPACES,
    ObjectCentricChunk,
)
from alexdoor_xas.recording import EpisodeBuffer, read_episode

# Step tables stacked into ``EpisodeRecord.obs``. ``obs_ref`` is skipped: its
# inline scalars duplicate proprio/object_state keys (see the dataset-contract wiki page).
_OBS_TABLES = ("proprio", "object_state", "contact")

CONTACT_FLAG_KEY = "contact_flag"
"""Virtual observation key: ``contact.sensed`` (force-sensing episodes) when
recorded, else the geometric ``contact.inferred`` fallback."""

DOOR_YAW_SIN_KEY = "door_yaw_sin"
DOOR_YAW_COS_KEY = "door_yaw_cos"
"""Virtual observation keys: sin/cos of the recorded per-step ``door_yaw_rad``
(smooth, wrap-free door-orientation encoding for pose-aware presets)."""

OBS_PRESETS: dict[str, tuple[str, ...]] = {
    # Available in every episode (phase2.v0 and v1): EE pose + door state, 9-dim.
    "core": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
    ),
    # core + best-available contact flag, 10-dim.
    "core_contact": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        CONTACT_FLAG_KEY,
    ),
    # Force-sensing Alex V2 episodes only: core + full joint state + contact
    # force. 2J + 11 dims (69 for the 29-joint Alex V2 model).
    "alex_full": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        "joint_pos",
        "joint_vel",
        "force_n",
        "sensed",
    ),
    # Door-pose-aware core (local post-Phase 3.3, additive): core + door-frame origin
    # relative to the robot base + sin/cos of the door yaw, 14-dim. Only valid
    # for episodes that recorded the door-pose object_state terms (the
    # stabilization data engine onward); older episodes fail with a clear
    # missing-key error. Extend with a hinge-origin block if door translation
    # variation ever needs richer pose context.
    "core_door_pose": (
        "ee_pos_w",
        "ee_quat_w_xyzw",
        "door_angle_rad",
        "door_angular_velocity_rad_s",
        "door_rel_pos_x",
        "door_rel_pos_y",
        "door_rel_pos_z",
        DOOR_YAW_SIN_KEY,
        DOOR_YAW_COS_KEY,
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


def discover_episodes(dataset_dir: str | Path) -> list[Path]:
    """Sorted episode HDF5 paths of one dataset version directory."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    return sorted(dataset_dir.glob("episode_*.hdf5"))


def read_dataset_meta(dataset_dir: str | Path) -> dict[str, Any]:
    """Parse the dataset-level ``meta.json`` written by the export pass."""
    meta_path = Path(dataset_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"dataset meta.json missing: {meta_path}")
    return json.loads(meta_path.read_text())


def load_episode_record(path: str | Path) -> EpisodeRecord:
    """Load one exported episode file into an :class:`EpisodeRecord`."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as h5:
        schema_version = str(h5.attrs.get("schema_version", ""))
    buffer = read_episode(path)
    if buffer.outcome is None:  # read_episode always sets it; keep types narrow
        raise ValueError(f"episode {path} has no outcome")

    obs: dict[str, np.ndarray] = {}
    if buffer.steps:
        for table in _OBS_TABLES:
            for key, first in getattr(buffer.steps[0], table).items():
                if isinstance(first, str):
                    continue  # tags like contact.source are not observations
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


def obs_dim(record: EpisodeRecord, preset: str = DEFAULT_OBS_PRESET) -> int:
    return int(obs_matrix(record, preset).shape[1])


def _preset_keys(preset: str) -> tuple[str, ...]:
    if preset not in OBS_PRESETS:
        raise ValueError(f"unknown obs preset {preset!r} (known: {sorted(OBS_PRESETS)})")
    return OBS_PRESETS[preset]


def _obs_key(record: EpisodeRecord, key: str) -> np.ndarray | None:
    if key == CONTACT_FLAG_KEY:
        return record.obs.get("sensed", record.obs.get("inferred"))
    if key in (DOOR_YAW_SIN_KEY, DOOR_YAW_COS_KEY):
        yaw = record.obs.get("door_yaw_rad")
        if yaw is None:
            return None
        return np.sin(yaw) if key == DOOR_YAW_SIN_KEY else np.cos(yaw)
    return record.obs.get(key)


class EpisodeDataset:
    """All episodes of one ``datasets/<task>/<action_space>/<version>/`` dir."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.meta = read_dataset_meta(self.dataset_dir)
        paths = discover_episodes(self.dataset_dir)
        if not paths:
            raise FileNotFoundError(f"no episode_*.hdf5 files in {self.dataset_dir}")
        self.records: list[EpisodeRecord] = [load_episode_record(p) for p in paths]

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

    def obs(self, index: int, preset: str = DEFAULT_OBS_PRESET) -> np.ndarray:
        return obs_matrix(self.records[index], preset)


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
        self.meta = read_dataset_meta(self.dataset_dir)
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
    episode_label = f"A4 episode {str(meta.get('episode_id', '<missing>'))[:8]}"
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
        episode_id=str(meta["episode_id"]),
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


def expected_action_space(dataset_dir: str | Path) -> str | None:
    """Action-space tag implied by the dataset layout (parent dir name), if any."""
    parent = Path(dataset_dir).resolve().parent.name
    return parent if parent in ALL_ACTION_SPACES else None


def open_dataset(dataset_dir: str | Path) -> EpisodeDataset | A4ChunkDataset:
    """Open a dataset dir with the right loader for its action space."""
    meta = read_dataset_meta(dataset_dir)
    if meta.get("action_space") == A4_OBJ_CENTRIC_CHUNK:
        return A4ChunkDataset(dataset_dir)
    return EpisodeDataset(dataset_dir)


__all__ = [
    "CONTACT_FLAG_KEY",
    "DEFAULT_OBS_PRESET",
    "OBS_PRESETS",
    "A4ChunkDataset",
    "A4EpisodeRecord",
    "EpisodeDataset",
    "EpisodeRecord",
    "discover_episodes",
    "expected_action_space",
    "load_episode_record",
    "obs_dim",
    "obs_matrix",
    "open_dataset",
    "read_dataset_meta",
]
