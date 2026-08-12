"""HDF5 + JSON-sidecar episode container (the Phase 2 storage choice).

Layout per episode (``episode_<id8>.hdf5`` + ``episode_<id8>.meta.json``)::

    /meta               attrs = EpisodeMeta fields
    /outcome            attrs = EpisodeOutcome fields
    /steps/t            (N,)
    /steps/action       (N, 6)         in meta.action_space
    /steps/<table>/<k>  (N, ...)       for obs_ref / proprio / object_state / contact / safety
    /extras/<name>      arrays (e.g. per-step door-frame actions, door frame pose)
    /extras_json        everything non-array (A4 chunk log, variation, ...)

HDF5 + JSON keeps episodes loadable by ACT/robomimic-style dataloaders in
Phase 3 while staying human-inspectable via the sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .episode import EpisodeBuffer, EpisodeMeta, EpisodeOutcome, EpisodeStep

SCHEMA_VERSION = "phase2.v2"
"""phase2.v2 replaces interpreted failure labels with factual termination data."""
LEGACY_SCHEMA_VERSIONS = ("phase2.v0", "phase2.v1")
_STEP_TABLES = ("obs_ref", "proprio", "object_state", "contact", "safety")


def episode_filename(meta: EpisodeMeta) -> str:
    return f"episode_{meta.episode_id[:8]}.hdf5"


def write_episode(buffer: EpisodeBuffer, directory: str | Path) -> Path:
    """Write one finished episode; returns the HDF5 path."""
    import h5py

    if buffer.outcome is None:
        raise ValueError("episode outcome must be set before writing")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h5_path = directory / episode_filename(buffer.meta)

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        meta_group = h5.create_group("meta")
        for key, value in buffer.meta.to_dict().items():
            meta_group.attrs[key] = value
        outcome_group = h5.create_group("outcome")
        for key, value in buffer.outcome.to_dict().items():
            outcome_group.attrs[key] = "" if value is None else value

        steps = h5.create_group("steps")
        steps.create_dataset("t", data=np.array([s.t for s in buffer.steps], dtype=np.float64))
        actions = (
            buffer.stacked(lambda s: s.action) if buffer.steps else np.zeros((0, EE_DELTA_DIM))
        )
        steps.create_dataset("action", data=actions)
        for table in _STEP_TABLES:
            _write_step_table(h5, steps, table, buffer.steps)

        extras = h5.create_group("extras")
        extras_json: dict[str, Any] = {}
        for name, value in buffer.extras.items():
            array = _as_array(value)
            if array is not None:
                extras.create_dataset(name, data=array)
            else:
                extras_json[name] = value
        h5.create_dataset("extras_json", data=json.dumps(extras_json))

    sidecar = h5_path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps({"meta": buffer.meta.to_dict(), "outcome": buffer.outcome.to_dict()}, indent=2)
        + "\n"
    )
    return h5_path


def read_episode(path: str | Path) -> EpisodeBuffer:
    """Read an episode written by :func:`write_episode` (full round-trip)."""
    import h5py

    with h5py.File(Path(path), "r") as h5:
        meta = EpisodeMeta(**{k: _from_h5(v) for k, v in h5["meta"].attrs.items()})
        schema_version = str(_from_h5(h5.attrs.get("schema_version", "")))
        outcome_raw = {k: _from_h5(v) for k, v in h5["outcome"].attrs.items()}
        # phase2.v0/v1 carried an interpreted failure_label. It is deliberately
        # ignored rather than exposed through the v2 model-facing API.
        outcome_raw.pop("failure_label", None)
        outcome_raw["success"] = bool(outcome_raw["success"])
        if schema_version in LEGACY_SCHEMA_VERSIONS:
            outcome_raw.setdefault("termination_reason", "not_recorded")
            outcome_raw.setdefault("environment_terminated", None)
            outcome_raw.setdefault("environment_truncated", None)
        else:
            outcome_raw["environment_terminated"] = _optional_bool(
                outcome_raw.get("environment_terminated")
            )
            outcome_raw["environment_truncated"] = _optional_bool(
                outcome_raw.get("environment_truncated")
            )
        outcome = EpisodeOutcome(**outcome_raw)

        steps_group = h5["steps"]
        t = np.asarray(steps_group["t"])
        action = np.asarray(steps_group["action"])
        tables = {table: _read_step_table(steps_group, table) for table in _STEP_TABLES}

        steps = [
            EpisodeStep(
                t=float(t[i]),
                action=action[i],
                obs_ref=tables["obs_ref"][i],
                proprio=tables["proprio"][i],
                object_state=tables["object_state"][i],
                contact=tables["contact"][i],
                safety=tables["safety"][i],
            )
            for i in range(len(t))
        ]

        extras: dict[str, Any] = json.loads(_from_h5(h5["extras_json"][()]))
        for name, dataset in h5["extras"].items():
            extras[name] = np.asarray(dataset)

    buffer = EpisodeBuffer(meta=meta, steps=steps, extras=extras)
    buffer.outcome = outcome
    return buffer


def _write_step_table(h5, steps_group, table: str, steps: list[EpisodeStep]) -> None:
    import h5py

    group = steps_group.create_group(table)
    if not steps:
        return
    for key in getattr(steps[0], table):
        values = [getattr(step, table)[key] for step in steps]
        first = values[0]
        if isinstance(first, str):
            group.create_dataset(key, data=values, dtype=h5py.string_dtype())
        elif isinstance(first, (bool, np.bool_)):
            group.create_dataset(key, data=np.array(values, dtype=bool))
        else:
            group.create_dataset(key, data=np.array(values, dtype=np.float64))


def _read_step_table(steps_group, table: str) -> list[dict[str, Any]]:
    group = steps_group[table]
    columns: dict[str, np.ndarray] = {key: np.asarray(group[key]) for key in group}
    n = max((len(col) for col in columns.values()), default=0)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {}
        for key, col in columns.items():
            value = col[i]
            if col.dtype == bool:
                row[key] = bool(value)
            elif col.dtype.kind in ("S", "O"):
                row[key] = value.decode() if isinstance(value, bytes) else str(value)
            elif col.ndim == 1:
                row[key] = float(value)
            else:
                row[key] = np.asarray(value)
        rows.append(row)
    return rows


def _as_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (dict, str)):
        try:
            return np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def _from_h5(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def _optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return bool(value)


__all__ = [
    "LEGACY_SCHEMA_VERSIONS",
    "SCHEMA_VERSION",
    "episode_filename",
    "read_episode",
    "write_episode",
]
