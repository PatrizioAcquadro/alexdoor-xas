"""Multi-action-space dataset export (datasets/<task>/<action_space>/<version>/).

From one set of recorded episodes the engine exports:

- **A2_ee_delta** — episodes as recorded (world-frame EE deltas).
- **A3_obj_rel_ee_delta** — the same episodes with per-step actions relabeled to
  the door-frame deltas stored in ``extras["action_door_frame"]`` at record time.
- **A4_obj_centric_chunk** — the controller's chunk log per episode, as JSON
  lines (struct/variable per docs/architecture.md).
- **A1_joint_delta** — only when every episode recorded per-tick joint targets
  (Alex episodes): per-step actions relabeled to full-body joint-position-target
  deltas. The proxy end-effector has no joints, so proxy runs skip A1.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)
from alexdoor_xas.dataset.robot_asset import dataset_robot_asset_payload
from alexdoor_xas.recording import EpisodeBuffer, write_episode


def export_datasets(
    episodes: list[EpisodeBuffer], datasets_root: str | Path, version: str = "v0"
) -> dict[str, Path]:
    """Export all Phase 2 action spaces; returns ``{action_space: dataset_dir}``.

    A dataset version directory is one generation pass: re-exporting the same
    version replaces it (episode ids are fresh UUIDs, so accumulating files
    across runs would silently corrupt the dataset).
    """
    if not episodes:
        raise ValueError("cannot export an empty episode list")
    # Validate the complete batch before replacing any existing dataset dirs.
    # This rejects mixed tasks and catches a V2 episode regardless of position.
    robot_asset = dataset_robot_asset_payload(episodes)
    task = episodes[0].meta.task
    root = Path(datasets_root)

    exported: dict[str, Path] = {}
    exported[A2_EE_DELTA] = _export_hdf5(
        episodes, root / task / A2_EE_DELTA / version, robot_asset
    )
    a3_episodes = [_relabel_to_door_frame(episode) for episode in episodes]
    exported[A3_OBJ_REL_EE_DELTA] = _export_hdf5(
        a3_episodes, root / task / A3_OBJ_REL_EE_DELTA / version, robot_asset
    )
    exported[A4_OBJ_CENTRIC_CHUNK] = _export_a4(
        episodes, root / task / A4_OBJ_CENTRIC_CHUNK / version, robot_asset
    )
    if all(_has_joint_targets(episode) for episode in episodes):
        a1_episodes = [_relabel_to_joint_delta(episode) for episode in episodes]
        exported[A1_JOINT_DELTA] = _export_hdf5(
            a1_episodes, root / task / A1_JOINT_DELTA / version, robot_asset
        )
    return exported


def _export_hdf5(
    episodes: list[EpisodeBuffer],
    directory: Path,
    robot_asset: dict | None,
) -> Path:
    _fresh_dir(directory)
    for episode in episodes:
        write_episode(episode, directory)
    _write_dataset_meta(episodes, directory, episodes[0].meta.action_space, robot_asset)
    return directory


def _export_a4(
    episodes: list[EpisodeBuffer],
    directory: Path,
    robot_asset: dict | None,
) -> Path:
    _fresh_dir(directory)
    lines = []
    for episode in episodes:
        meta = dataclasses.replace(episode.meta, action_space=A4_OBJ_CENTRIC_CHUNK)
        record = {
            "meta": meta.to_dict(),
            "chunks": episode.extras.get("a4_chunks", []),
            "outcome": episode.outcome.to_dict() if episode.outcome else None,
        }
        lines.append(json.dumps(record))
    (directory / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    _write_dataset_meta(episodes, directory, A4_OBJ_CENTRIC_CHUNK, robot_asset)
    return directory


def _relabel_to_door_frame(episode: EpisodeBuffer) -> EpisodeBuffer:
    actions_door = np.asarray(episode.extras["action_door_frame"], dtype=np.float64)
    if actions_door.shape[0] != episode.n_steps:
        raise ValueError(
            f"episode {episode.meta.episode_id} has {episode.n_steps} steps but "
            f"{actions_door.shape[0]} door-frame actions"
        )
    steps = [
        dataclasses.replace(step, action=actions_door[i])
        for i, step in enumerate(episode.steps)
    ]
    relabeled = EpisodeBuffer(
        meta=dataclasses.replace(episode.meta, action_space=A3_OBJ_REL_EE_DELTA),
        steps=steps,
        extras=dict(episode.extras),
    )
    relabeled.outcome = episode.outcome
    return relabeled


def _has_joint_targets(episode: EpisodeBuffer) -> bool:
    return bool(episode.steps) and "joint_pos_target" in episode.steps[0].proprio


def _relabel_to_joint_delta(episode: EpisodeBuffer) -> EpisodeBuffer:
    """Relabel per-step actions to full-body joint-position-target deltas (A1).

    ``proprio.joint_pos_target[t]`` is captured pre-step (the target applied
    during tick t-1), so the target change produced by ``action[t]`` is
    ``target[t+1] - target[t]``. The final step uses the post-loop applied
    target recorded in ``extras["final_joint_pos_target"]``; episodes recorded
    before that extra existed fall back to a zero last delta (the scripted
    controller's terminal hold command is ~zero anyway).
    """
    targets = np.stack(
        [np.asarray(step.proprio["joint_pos_target"], dtype=np.float64) for step in episode.steps]
    )
    final_target = episode.extras.get("final_joint_pos_target")
    last = (
        np.asarray(final_target, dtype=np.float64).reshape(1, -1)
        if final_target is not None
        else targets[-1:].copy()
    )
    deltas = np.diff(np.concatenate([targets, last], axis=0), axis=0)
    steps = [
        dataclasses.replace(step, action=deltas[i]) for i, step in enumerate(episode.steps)
    ]
    relabeled = EpisodeBuffer(
        meta=dataclasses.replace(episode.meta, action_space=A1_JOINT_DELTA),
        steps=steps,
        extras=dict(episode.extras),
    )
    relabeled.outcome = episode.outcome
    return relabeled


def _write_dataset_meta(
    episodes: list[EpisodeBuffer],
    directory: Path,
    action_space: str,
    robot_asset: dict | None,
) -> None:
    outcomes = [episode.outcome for episode in episodes if episode.outcome is not None]
    meta = {
        "task": episodes[0].meta.task,
        "action_space": action_space,
        "n_episodes": len(episodes),
        "n_success": sum(1 for outcome in outcomes if outcome.success),
        "seeds": [episode.meta.seed for episode in episodes],
        "robot": episodes[0].meta.robot,
        "scene": episodes[0].meta.scene,
        "policy": episodes[0].meta.policy,
        "generator": {
            "engine_cfg": episodes[0].extras.get("engine_cfg"),
            "controller_cfg": episodes[0].extras.get("controller_cfg"),
        },
        "robot_asset": robot_asset,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _fresh_dir(directory: Path) -> None:
    import shutil

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


__all__ = ["export_datasets"]
