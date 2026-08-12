"""Deterministic translation-only open-loop evaluation and one summary figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.dataset import EpisodeRecord, obs_matrix

TRANSLATION_LABELS = ("dx", "dy", "dz")


def predict_episode_open_loop(
    policy, record: EpisodeRecord, stride: int | None = None
) -> np.ndarray:
    """Predict an action at every recorded step by stitching deterministic chunks."""
    stride = policy.chunk_size if stride is None else int(stride)
    if not 1 <= stride <= policy.chunk_size:
        raise ValueError(f"stride must be in [1, {policy.chunk_size}], got {stride}")
    observations = obs_matrix(record, policy.obs_preset)
    predicted = np.zeros((record.n_steps, policy.stats.action.dim), dtype=np.float64)
    for start in range(0, record.n_steps, stride):
        chunk = policy.predict(observations[start])
        stop = min(start + stride, record.n_steps)
        predicted[start:stop] = chunk[: stop - start]
    return predicted


def open_loop_report(
    policy,
    records: list[EpisodeRecord],
    json_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    """Write the compact translation-only metrics and one worst-episode plot."""
    if not records:
        raise ValueError("open-loop report needs at least one episode record")

    evaluated: list[tuple[EpisodeRecord, np.ndarray, float]] = []
    absolute_error_sum = np.zeros(3, dtype=np.float64)
    evaluated_steps = 0
    per_episode: list[dict[str, Any]] = []
    for record in records:
        if record.actions.ndim != 2 or record.actions.shape[1] < 3:
            raise ValueError(f"episode {record.episode_id} has no dx/dy/dz action block")
        predicted = predict_episode_open_loop(policy, record, stride=stride)
        absolute_error = np.abs(predicted[:, :3] - record.actions[:, :3])
        l1_mean = float(absolute_error.mean())
        absolute_error_sum += absolute_error.sum(axis=0)
        evaluated_steps += record.n_steps
        per_episode.append(
            {
                "episode_id": record.episode_id,
                "l1_mean": l1_mean,
                "evaluated_steps": record.n_steps,
            }
        )
        evaluated.append((record, predicted, l1_mean))

    representative = sorted(evaluated, key=lambda item: (-item[2], item[0].episode_id))[0]
    l1_by_dimension = absolute_error_sum / evaluated_steps
    report = {
        "action_space": policy.action_space,
        "obs_preset": policy.obs_preset,
        "stride": policy.chunk_size if stride is None else int(stride),
        "aggregate_l1_mean": float(l1_by_dimension.mean()),
        "l1_by_dimension": {
            label: float(value)
            for label, value in zip(TRANSLATION_LABELS, l1_by_dimension, strict=True)
        },
        "per_episode": per_episode,
        "evaluated_steps": evaluated_steps,
        "representative_episode_id": representative[0].episode_id,
        "representative_selection": "worst translation L1; episode ID ascending tie-break",
    }
    if json_path is not None:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if summary_path is not None:
        _open_loop_summary(
            representative[0], representative[1], l1_by_dimension, Path(summary_path)
        )
    return report


def _open_loop_summary(
    record: EpisodeRecord,
    predicted: np.ndarray,
    l1_by_dimension: np.ndarray,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6.5))
    grid = fig.add_gridspec(2, 3)
    error_axis = fig.add_subplot(grid[0, :])
    error_axis.bar(TRANSLATION_LABELS, l1_by_dimension)
    error_axis.set_ylabel("L1")
    error_axis.set_title(f"Open-loop translation error — {record.episode_id}")
    error_axis.grid(axis="y", alpha=0.25)
    for dim, label in enumerate(TRANSLATION_LABELS):
        axis = fig.add_subplot(grid[1, dim])
        axis.plot(record.actions[:, dim], label="recorded", linewidth=1.1)
        axis.plot(predicted[:, dim], label="predicted", linewidth=1.1, linestyle="--")
        axis.set_title(label)
        axis.set_xlabel("step")
        axis.grid(alpha=0.2)
        if dim == 0:
            axis.set_ylabel("action")
            axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = ["open_loop_report", "predict_episode_open_loop"]
