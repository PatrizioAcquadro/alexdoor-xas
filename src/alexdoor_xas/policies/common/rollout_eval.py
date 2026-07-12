"""Pure helpers for chunk-policy closed-loop evaluation summaries."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np

DETERMINISM_TOLERANCES = {
    "command_abs": 1e-9,
    "angle_abs_rad": 1e-9,
    "force_abs_n": 1e-6,
}
"""Repeat-same-seed comparison tolerances. Headless physics is deterministic
in this build and the policy sampling generator is reseeded identically, so
repeats are expected to be bit-identical; the tolerances only absorb
float32<->float64 round-trips in the readers."""


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
    termination_reason: str = "",
) -> str | None:
    """Coarse per-rollout failure taxonomy (None on success).

    Mirrors the data-engine convention of labeling every non-success; kept
    deliberately coarse — the later unified report needs stable buckets, not
    per-run prose. ``termination_reason`` (``RolloutResult.termination_reason``)
    disambiguates env truncation from a plain tick-budget timeout.
    """
    if success:
        return None
    # Rejections take precedence over no_contact: a rejection storm executes
    # zero motion and therefore zero contact — labeling it no_contact would
    # misdiagnose an adapter/frame problem as a policy-reach problem.
    if termination_reason == "rejection_stop" or "rejected" in notes:
        return "stopped_on_rejection"
    if n_rejected > 0:
        return "commands_rejected"
    if termination_reason == "env_truncated":
        return "env_truncated"
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
        # Across-seed spread of the fixed-reset block (randomization disabled,
        # *different* reset/sampling seeds). This is output variability across
        # seeds — never determinism evidence; the repeat-same-seed probe
        # (determinism_probe) carries the actual determinism claim.
        "fixed_reset_spread_rad": (
            max(fixed_finals) - min(fixed_finals) if fixed_finals else None
        ),
    }


def _rollout_traces(result) -> dict[str, Any]:
    """Comparable trace arrays/sequences of one ``RolloutResult``."""
    requested = [
        np.zeros(6) if d.requested is None else np.asarray(d.requested, dtype=np.float64)
        for d in result.decisions_per_tick
    ]
    applied = [
        np.zeros(6) if d.applied is None else np.asarray(d.applied, dtype=np.float64)
        for d in result.decisions_per_tick
    ]
    return {
        "n_ticks": int(result.n_ticks),
        "requested": np.stack(requested) if requested else np.zeros((0, 6)),
        "applied": np.stack(applied) if applied else np.zeros((0, 6)),
        "statuses": [str(d.status) for d in result.decisions_per_tick],
        "first_success_tick": result.first_success_tick,
        "termination_reason": result.termination_reason,
        "final_angle_rad": float(result.final_angle_rad),
        "contact": [None if c is None else bool(c) for c in result.contact_per_tick],
        "force": [None if f is None else float(f) for f in result.force_n_per_tick],
    }


def rollout_trace_hash(result) -> str:
    """sha256 over one rollout's command/state traces (exact bytes)."""
    traces = _rollout_traces(result)
    digest = hashlib.sha256()
    digest.update(str(traces["n_ticks"]).encode())
    digest.update(traces["requested"].tobytes())
    digest.update(traces["applied"].tobytes())
    digest.update("|".join(traces["statuses"]).encode())
    digest.update(str(traces["first_success_tick"]).encode())
    digest.update(traces["termination_reason"].encode())
    digest.update(np.float64(traces["final_angle_rad"]).tobytes())
    digest.update("|".join(str(c) for c in traces["contact"]).encode())
    digest.update(
        np.asarray(
            [np.nan if f is None else f for f in traces["force"]], dtype=np.float64
        ).tobytes()
    )
    return digest.hexdigest()


def determinism_probe_report(
    results: list,
    *,
    seed: int,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Genuine repeat-same-seed determinism evidence for one rollout config.

    ``results`` are >= 2 ``RolloutResult`` runs of the *same* environment
    reset seed, policy sampling seed, pose, checkpoint, and configuration.
    Every repeat is compared to the first: tick counts, per-tick
    requested/adapted command traces, adapter decision statuses, the success
    crossing tick, termination reason, final state, and force/contact traces,
    within explicit tolerances. This — not the across-seed ``fixed_reset``
    spread — is the determinism claim.
    """
    if len(results) < 2:
        raise ValueError("determinism probe needs at least 2 repeat rollouts")
    tolerances = dict(DETERMINISM_TOLERANCES if tolerances is None else tolerances)
    reference = _rollout_traces(results[0])
    mismatches: list[str] = []
    max_diffs = {"requested": 0.0, "applied": 0.0, "final_angle_rad": 0.0, "force_n": 0.0}

    for index, result in enumerate(results[1:], start=1):
        label = f"repeat {index}"
        traces = _rollout_traces(result)
        if traces["n_ticks"] != reference["n_ticks"]:
            mismatches.append(
                f"{label}: n_ticks {traces['n_ticks']} != {reference['n_ticks']}"
            )
            continue  # trace lengths differ; elementwise comparison is meaningless
        for key in ("first_success_tick", "termination_reason"):
            if traces[key] != reference[key]:
                mismatches.append(f"{label}: {key} {traces[key]!r} != {reference[key]!r}")
        if traces["statuses"] != reference["statuses"]:
            mismatches.append(f"{label}: adapter decision statuses differ")
        if traces["contact"] != reference["contact"]:
            mismatches.append(f"{label}: contact trace differs")
        for key, tol_name in (("requested", "command_abs"), ("applied", "command_abs")):
            diff = (
                float(np.max(np.abs(traces[key] - reference[key])))
                if reference[key].size
                else 0.0
            )
            max_diffs[key] = max(max_diffs[key], diff)
            if diff > tolerances[tol_name]:
                mismatches.append(
                    f"{label}: {key} command trace differs by {diff:.3g} "
                    f"(> {tolerances[tol_name]:.3g})"
                )
        angle_diff = abs(traces["final_angle_rad"] - reference["final_angle_rad"])
        max_diffs["final_angle_rad"] = max(max_diffs["final_angle_rad"], angle_diff)
        if angle_diff > tolerances["angle_abs_rad"]:
            mismatches.append(
                f"{label}: final angle differs by {angle_diff:.3g} rad "
                f"(> {tolerances['angle_abs_rad']:.3g})"
            )
        force_pairs = [
            (a, b)
            for a, b in zip(traces["force"], reference["force"], strict=True)
            if a is not None and b is not None
        ]
        force_diff = max((abs(a - b) for a, b in force_pairs), default=0.0)
        max_diffs["force_n"] = max(max_diffs["force_n"], force_diff)
        if force_diff > tolerances["force_abs_n"]:
            mismatches.append(
                f"{label}: force trace differs by {force_diff:.3g} N "
                f"(> {tolerances['force_abs_n']:.3g})"
            )

    return {
        "kind": "repeat_same_seed",
        "seed": seed,
        "repeats": len(results),
        "tolerances": tolerances,
        "trace_sha256": [rollout_trace_hash(result) for result in results],
        "max_abs_diffs": max_diffs,
        "mismatches": mismatches,
        "passed": not mismatches,
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
    "DETERMINISM_TOLERANCES",
    "aggregate_rollout_rows",
    "contact_report",
    "determinism_probe_report",
    "rollout_failure_label",
    "rollout_trace_hash",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
]
