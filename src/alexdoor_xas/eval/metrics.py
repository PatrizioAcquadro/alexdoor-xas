"""Operational metrics for recorded episodes."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from alexdoor_xas.recording import EpisodeBuffer


def episode_metrics(episode: EpisodeBuffer) -> dict[str, Any]:
    """Compute the scalar metrics used by scripted runs."""
    if episode.outcome is None:
        raise ValueError("episode outcome must be set before computing metrics")

    angles = np.array(
        [step.object_state["door_angle_rad"] for step in episode.steps], dtype=np.float64
    )
    times = np.array([step.t for step in episode.steps], dtype=np.float64)
    contact = np.array([bool(step.contact["sensed"]) for step in episode.steps], dtype=bool)
    forces = np.array([float(step.contact["force_n"]) for step in episode.steps], dtype=np.float64)
    sensed_forces = forces[contact]
    success_angle = _success_angle(episode)
    time_to_threshold = None
    if angles.size and success_angle is not None:
        reached = np.nonzero(angles >= success_angle)[0]
        if reached.size:
            time_to_threshold = float(times[reached[0]])

    return {
        "episode_id": episode.meta.episode_id,
        "seed": episode.meta.seed,
        "randomized": episode.extras.get("variation") is not None,
        "success": episode.outcome.success,
        "termination_reason": episode.outcome.termination_reason,
        "environment_terminated": episode.outcome.environment_terminated,
        "environment_truncated": episode.outcome.environment_truncated,
        "final_door_angle_rad": episode.outcome.final_door_angle,
        "max_door_angle_rad": float(angles.max()) if angles.size else 0.0,
        "time_to_threshold_s": time_to_threshold,
        "n_steps": episode.n_steps,
        "duration_s": float(times[-1] + episode.meta.control_dt) if episode.n_steps else 0.0,
        "contact_ticks": int(contact.sum()),
        "mean_contact_force_n": float(sensed_forces.mean()) if sensed_forces.size else None,
        "max_contact_force_n": float(sensed_forces.max()) if sensed_forces.size else None,
    }


def aggregate_metrics(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one scripted run."""
    if not per_episode:
        return {"n_episodes": 0}
    finals = np.array([m["final_door_angle_rad"] for m in per_episode], dtype=np.float64)
    finite_finals = finals[np.isfinite(finals)]
    successes = [m for m in per_episode if m["success"]]
    times = [m["time_to_threshold_s"] for m in successes if m["time_to_threshold_s"] is not None]
    termination_counts = Counter(m["termination_reason"] for m in per_episode)
    summary = {
        "n_episodes": len(per_episode),
        "n_fixed": sum(1 for m in per_episode if not m["randomized"]),
        "n_randomized": sum(1 for m in per_episode if m["randomized"]),
        "n_success": len(successes),
        "success_rate": len(successes) / len(per_episode),
        "final_door_angle_rad": {
            "mean": float(finite_finals.mean()) if finite_finals.size else math.nan,
            "min": float(finite_finals.min()) if finite_finals.size else math.nan,
            "max": float(finite_finals.max()) if finite_finals.size else math.nan,
        },
        "mean_time_to_threshold_s": float(np.mean(times)) if times else None,
        "termination_reasons": dict(termination_counts),
    }
    force_eps = [m for m in per_episode if m["mean_contact_force_n"] is not None]
    if force_eps:
        summary["contact_force_n"] = {
            "mean_of_means": float(np.mean([m["mean_contact_force_n"] for m in force_eps])),
            "max": float(np.max([m["max_contact_force_n"] for m in force_eps])),
            "mean_contact_ticks": float(np.mean([m["contact_ticks"] for m in force_eps])),
        }
    return summary


def _success_angle(episode: EpisodeBuffer) -> float | None:
    engine_cfg = episode.extras.get("engine_cfg")
    if isinstance(engine_cfg, dict) and "success_angle_rad" in engine_cfg:
        return float(engine_cfg["success_angle_rad"])
    return None


__all__ = ["aggregate_metrics", "episode_metrics"]
