"""Pure helpers for ACT closed-loop evaluation summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any


def seed_protocol(
    *,
    base_seed: int,
    episodes_fixed: int,
    episodes_randomized: int,
    variation_bounds,
) -> dict[str, Any]:
    """Describe the fixed/randomized seed plan shared by ACT and references."""
    fixed = [base_seed + i for i in range(episodes_fixed)]
    randomized_start = base_seed + episodes_fixed
    randomized = [randomized_start + i for i in range(episodes_randomized)]
    return {
        "base_seed": base_seed,
        "episodes_fixed": episodes_fixed,
        "episodes_randomized": episodes_randomized,
        "fixed_seeds": fixed,
        "randomized_seeds": randomized,
        "variation_bounds": asdict(variation_bounds),
    }


def summarize_decision_warnings(decisions) -> dict[str, Any]:
    """Count adapter warnings in an ordered decision sequence."""
    counts: Counter[str] = Counter()
    for decision in decisions:
        for warning in getattr(decision, "warnings", ()) or ():
            counts[str(warning)] += 1
    return {
        "n_warnings": sum(counts.values()),
        "warning_counts": dict(sorted(counts.items())),
    }


def aggregate_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ACT rollout rows while preserving the existing adapter fields."""
    if not rows:
        raise ValueError("cannot aggregate zero rollout rows")

    finals = [float(row["final_angle_rad"]) for row in rows]
    fixed_finals = [float(row["final_angle_rad"]) for row in rows if not row["randomized"]]
    warning_counts: Counter[str] = Counter()
    for row in rows:
        warning_counts.update(row.get("warning_counts", {}))

    return {
        "n_rollouts": len(rows),
        "n_fixed": sum(1 for row in rows if not row["randomized"]),
        "n_randomized": sum(1 for row in rows if row["randomized"]),
        "n_success": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "final_angle_rad": {
            "mean": sum(finals) / len(finals),
            "min": min(finals),
            "max": max(finals),
        },
        "mean_ticks": sum(float(row["n_ticks"]) for row in rows) / len(rows),
        "adapter": {
            "n_accepted": sum(int(row["n_accepted"]) for row in rows),
            "n_corrected": sum(int(row["n_corrected"]) for row in rows),
            "n_rejected": sum(int(row["n_rejected"]) for row in rows),
            "n_warnings": sum(int(row.get("n_warnings", 0)) for row in rows),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "fixed_determinism_spread_rad": (
            max(fixed_finals) - min(fixed_finals) if fixed_finals else None
        ),
    }


def scripted_reference_payload(
    *,
    per_episode_metrics: list[dict[str, Any]],
    aggregate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """JSON block for the opt-in matched scripted reference."""
    return {
        "enabled": True,
        "seed_protocol": protocol,
        "episode_counts": {
            "n_episodes": aggregate.get("n_episodes", len(per_episode_metrics)),
            "n_fixed": aggregate.get("n_fixed"),
            "n_randomized": aggregate.get("n_randomized"),
        },
        "aggregate": aggregate,
        "episodes": per_episode_metrics,
    }


__all__ = [
    "aggregate_rollout_rows",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
]
