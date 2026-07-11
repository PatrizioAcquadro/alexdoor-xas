"""Pure helpers for chunk-policy closed-loop evaluation summaries."""

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


def contact_report(
    contact_per_tick: list[bool | None],
    force_n_per_tick: list[float | None],
    control_dt: float,
    admission_bound_n: float | None = None,
) -> dict[str, Any]:
    """Per-rollout contact/force summary from the rollout's per-tick capture.

    ``contact_ticks`` counts force-sensed contact ticks; force stats
    (mean/max/p95, newtons) are over contact ticks only, and ``impulse_ns`` is
    the |force|·dt sum over the whole rollout. When the env exposed no force
    sensing the summary says so explicitly instead of reporting zeros.
    """
    # One contact_ticks definition for both branches: the sensed-contact flag
    # count (force presence only gates the force *statistics*).
    contact_ticks = sum(1 for c in contact_per_tick if c)
    forces = [f for f in force_n_per_tick if f is not None]
    if not forces:
        return {
            "contact_ticks": contact_ticks,
            "contact_source": None,
            "force_n": None,
            "impulse_ns": None,
            "force_exceeds_admission_bound": None,
            "unavailable_reason": "env exposes no contact force sensing",
        }
    contact_forces = sorted(
        f
        for f, c in zip(force_n_per_tick, contact_per_tick, strict=True)
        if f is not None and c
    )
    if contact_forces:
        p95_index = min(len(contact_forces) - 1, int(round(0.95 * (len(contact_forces) - 1))))
        force_stats = {
            "mean": sum(contact_forces) / len(contact_forces),
            "max": contact_forces[-1],
            "p95": contact_forces[p95_index],
        }
    else:
        force_stats = {"mean": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "contact_ticks": contact_ticks,
        "contact_source": "force_sensor",
        "force_n": force_stats,
        "impulse_ns": sum(forces) * control_dt,
        # Watch-item flag: the dataset admission policy bounds *recorded* demo
        # forces; a learned policy exceeding the bound is not an eval failure,
        # but the unified report must be able to see it (None = no bound given).
        "force_exceeds_admission_bound": (
            force_stats["max"] > admission_bound_n if admission_bound_n is not None else None
        ),
        "unavailable_reason": None,
    }


def rollout_failure_label(
    *,
    success: bool,
    n_ticks: int,
    max_ticks: int,
    contact_ticks: int,
    n_rejected: int,
    notes: str,
) -> str | None:
    """Coarse per-rollout failure taxonomy (None on success).

    Mirrors the data-engine convention of labeling every non-success; kept
    deliberately coarse — the later unified report needs stable buckets, not
    per-run prose.
    """
    if success:
        return None
    # Rejections take precedence over no_contact: a rejection storm executes
    # zero motion and therefore zero contact — labeling it no_contact would
    # misdiagnose an adapter/frame problem as a policy-reach problem.
    if "rejected" in notes:
        return "stopped_on_rejection"
    if n_rejected > 0:
        return "commands_rejected"
    if contact_ticks == 0:
        return "no_contact"
    if n_ticks >= max_ticks:
        return "timeout_no_success"
    return "policy_stopped_early"


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
    "contact_report",
    "rollout_failure_label",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
]
