"""Open-loop prediction inspection: predicted vs. recorded action chunks.

Stitches z=0 chunk predictions along each episode's recorded observations
(no environment involved) and reports per-dimension L1/MSE — the cheap sanity
check between offline loss and closed-loop rollout. Matplotlib is lazy and
optional (Agg backend), matching ``eval/plots.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.dataset import EpisodeRecord, obs_matrix
from alexdoor_xas.policies.act.policy import ActPolicy


def predict_episode_open_loop(policy: ActPolicy, record: EpisodeRecord) -> np.ndarray:
    """Predicted actions ``(N, D)``: chunks queried at t = 0, H, 2H, ... ."""
    observations = obs_matrix(record, policy.obs_preset)
    n_steps = record.n_steps
    predicted = np.zeros((n_steps, policy.stats.action.dim), dtype=np.float64)
    for start in range(0, n_steps, policy.chunk_size):
        chunk = policy.predict(observations[start])
        stop = min(start + policy.chunk_size, n_steps)
        predicted[start:stop] = chunk[: stop - start]
    return predicted


def open_loop_report(
    policy: ActPolicy,
    records: list[EpisodeRecord],
    json_path: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Per-episode and aggregate prediction errors; optionally saved to disk."""
    if not records:
        raise ValueError("open-loop report needs at least one episode record")

    episodes: dict[str, Any] = {}
    abs_error_sum: np.ndarray | None = None
    sq_error_sum: np.ndarray | None = None
    total_steps = 0
    for record in records:
        predicted = predict_episode_open_loop(policy, record)
        error = predicted - record.actions
        abs_error = np.abs(error)
        episodes[record.episode_id] = {
            "n_steps": record.n_steps,
            "l1_per_dim": abs_error.mean(axis=0).tolist(),
            "mse_per_dim": (error**2).mean(axis=0).tolist(),
            "l1_mean": float(abs_error.mean()),
        }
        abs_error_sum = abs_error.sum(axis=0) + (0 if abs_error_sum is None else abs_error_sum)
        sq_error_sum = (error**2).sum(axis=0) + (0 if sq_error_sum is None else sq_error_sum)
        total_steps += record.n_steps
        if plots_dir is not None:
            _open_loop_plot(
                record,
                predicted,
                Path(plots_dir) / f"open_loop_{record.episode_id[:8]}.png",
            )

    report = {
        "action_space": policy.action_space,
        "obs_preset": policy.obs_preset,
        "chunk_size": policy.chunk_size,
        "n_episodes": len(records),
        "episodes": episodes,
        "aggregate": {
            "l1_per_dim": (abs_error_sum / total_steps).tolist(),
            "mse_per_dim": (sq_error_sum / total_steps).tolist(),
            "l1_mean": float(abs_error_sum.sum() / (total_steps * abs_error_sum.shape[0])),
            "n_steps": total_steps,
        },
    }
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _open_loop_plot(record: EpisodeRecord, predicted: np.ndarray, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels = ("dx (m)", "dy (m)", "dz (m)")
    for dim, (ax, label) in enumerate(zip(axes, labels, strict=True)):
        ax.plot(record.actions[:, dim], lw=1.2, label="recorded")
        ax.plot(predicted[:, dim], lw=1.2, ls="--", label="predicted")
        ax.set_ylabel(label)
    axes[0].set_title(
        f"Open-loop ACT prediction — episode {record.episode_id[:8]} "
        f"({record.action_space}; rotation dims constant zero, omitted)"
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("tick")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = ["open_loop_report", "predict_episode_open_loop"]
