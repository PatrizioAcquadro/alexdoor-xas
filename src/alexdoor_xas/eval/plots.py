"""Phase 2 plots: door angle vs. time and final-angle summary (matplotlib/Agg)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from alexdoor_xas.recording import EpisodeBuffer


def door_angle_plot(episodes: list[EpisodeBuffer], path: str | Path) -> Path:
    """Overlay door angle vs. time for all episodes, with the success threshold."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    threshold = None
    for episode in episodes:
        times = [step.t for step in episode.steps]
        angles = [step.object_state["door_angle_rad"] for step in episode.steps]
        label = f"seed {episode.meta.seed}" + (
            " (rand)" if episode.extras.get("variation") is not None else ""
        )
        ax.plot(times, np.degrees(angles), lw=1.5, label=label)
        engine_cfg = episode.extras.get("engine_cfg") or {}
        threshold = engine_cfg.get("success_angle_rad", threshold)

    if threshold is not None:
        ax.axhline(
            math.degrees(threshold), color="0.4", ls="--", lw=1.0, label="success threshold"
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("door angle (deg)")
    ax.set_title("Scripted door push: door angle vs. time")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def final_angle_plot(episodes: list[EpisodeBuffer], path: str | Path) -> Path:
    """Final door angle per episode, colored by success."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels = [f"{episode.meta.seed}" for episode in episodes]
    finals = [
        math.degrees(episode.outcome.final_door_angle) if episode.outcome else 0.0
        for episode in episodes
    ]
    colors = [
        "tab:green" if episode.outcome and episode.outcome.success else "tab:red"
        for episode in episodes
    ]
    ax.bar(range(len(episodes)), finals, color=colors)
    ax.set_xticks(range(len(episodes)), labels)
    ax.set_xlabel("episode seed")
    ax.set_ylabel("final door angle (deg)")
    ax.set_title("Scripted door push: final door angle per episode")
    return _save(fig, path)


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


__all__ = ["door_angle_plot", "final_angle_plot"]
