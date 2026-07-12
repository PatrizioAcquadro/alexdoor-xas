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

    ``contact_ticks`` counts force-sensed contact ticks; ``force_n`` stats
    (mean/max/p95, newtons) are over contact ticks only. Admission evidence in
    ``force_n_all_samples`` inspects every sensed sample independently of the
    contact classifier. ``impulse_ns`` is the |force|·dt sum over the whole
    rollout. When the env exposed no force sensing the summary says so
    explicitly instead of reporting zeros.
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
            "force_n_all_samples": None,
            "impulse_ns": None,
            "force_exceeds_admission_bound": None,
            "unavailable_reason": "env exposes no contact force sensing",
        }
    if not np.isfinite(forces).all():
        raise RuntimeError("non-finite rollout force trace cannot become evidence")
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
    n_exceedance_ticks = (
        sum(force > admission_bound_n for force in forces)
        if admission_bound_n is not None
        else None
    )
    return {
        "contact_ticks": contact_ticks,
        "contact_source": "force_sensor",
        "force_n": force_stats,
        "force_n_all_samples": {
            "max": max(forces),
            "n_exceedance_ticks": n_exceedance_ticks,
        },
        "impulse_ns": sum(forces) * control_dt,
        # Watch-item flag: the dataset admission policy bounds *recorded* demo
        # forces; a learned policy exceeding the bound is not an eval failure,
        # but the unified report must be able to see it (None = no bound given).
        "force_exceeds_admission_bound": (
            n_exceedance_ticks > 0 if n_exceedance_ticks is not None else None
        ),
        "unavailable_reason": None,
    }


def force_trace_evidence(
    result,
    *,
    admission_bound_n: float,
    window_radius: int = 2,
) -> dict[str, Any] | None:
    """Bind force peaks to their per-tick contact, command, and adapter trace."""
    available = [
        (index, float(force))
        for index, force in enumerate(result.force_n_per_tick)
        if force is not None
    ]
    if not available:
        return None
    if not np.isfinite([force for _, force in available]).all():
        raise RuntimeError("non-finite rollout force trace cannot become evidence")
    peak_index, peak_force = max(available, key=lambda item: item[1])
    exceedance_indices = [index for index, force in available if force > admission_bound_n]

    def sample(index: int) -> dict[str, Any]:
        decision = result.decisions_per_tick[index]
        return {
            "tick": index + 1,
            "force_n": result.force_n_per_tick[index],
            "contact": result.contact_per_tick[index],
            "status": str(decision.status),
            "requested": (
                None
                if decision.requested is None
                else np.asarray(decision.requested, dtype=np.float64).tolist()
            ),
            "applied": (
                None
                if decision.applied is None
                else np.asarray(decision.applied, dtype=np.float64).tolist()
            ),
        }

    decision = result.decisions_per_tick[peak_index]
    start = max(0, peak_index - window_radius)
    stop = min(len(result.force_n_per_tick), peak_index + window_radius + 1)
    return {
        "trace_sha256": rollout_trace_hash(result),
        "admission_bound_n": float(admission_bound_n),
        "peak_tick": peak_index + 1,
        "peak_force_n": peak_force,
        "peak_contact": result.contact_per_tick[peak_index],
        "peak_status": str(decision.status),
        "peak_requested": (
            None
            if decision.requested is None
            else np.asarray(decision.requested, dtype=np.float64).tolist()
        ),
        "peak_applied": (
            None
            if decision.applied is None
            else np.asarray(decision.applied, dtype=np.float64).tolist()
        ),
        "n_exceedance_ticks": len(exceedance_indices),
        "exceedance_ticks": [index + 1 for index in exceedance_indices],
        "window": [sample(index) for index in range(start, stop)],
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
    if termination_reason == "invalid_simulator_state":
        return "invalid_simulator_state"
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


def rollout_traces_payload(result) -> dict[str, Any]:
    """JSON-able trace record of one ``RolloutResult`` (determinism evidence).

    Everything the repeat-same-seed comparison needs: per-tick
    requested/adapted commands, adapter decision statuses, contact/force
    traces, the success crossing tick, termination reason, and final state.
    """
    requested = [
        [0.0] * 6 if d.requested is None else np.asarray(d.requested, dtype=np.float64).tolist()
        for d in result.decisions_per_tick
    ]
    applied = [
        [0.0] * 6 if d.applied is None else np.asarray(d.applied, dtype=np.float64).tolist()
        for d in result.decisions_per_tick
    ]
    return {
        "n_ticks": int(result.n_ticks),
        "requested": requested,
        "applied": applied,
        "statuses": [str(d.status) for d in result.decisions_per_tick],
        "first_success_tick": result.first_success_tick,
        "termination_reason": result.termination_reason,
        "final_angle_rad": float(result.final_angle_rad),
        "contact": [None if c is None else bool(c) for c in result.contact_per_tick],
        "force": [None if f is None else float(f) for f in result.force_n_per_tick],
    }


def trace_payload_hash(traces: dict[str, Any]) -> str:
    """sha256 over one rollout's command/state traces (exact bytes)."""
    digest = hashlib.sha256()
    digest.update(str(traces["n_ticks"]).encode())
    digest.update(np.asarray(traces["requested"], dtype=np.float64).tobytes())
    digest.update(np.asarray(traces["applied"], dtype=np.float64).tobytes())
    digest.update("|".join(traces["statuses"]).encode())
    digest.update(str(traces["first_success_tick"]).encode())
    digest.update(str(traces["termination_reason"]).encode())
    digest.update(np.float64(traces["final_angle_rad"]).tobytes())
    digest.update("|".join(str(c) for c in traces["contact"]).encode())
    digest.update(
        np.asarray(
            [np.nan if f is None else f for f in traces["force"]], dtype=np.float64
        ).tobytes()
    )
    return digest.hexdigest()


def rollout_trace_hash(result) -> str:
    """sha256 over one rollout's command/state traces (exact bytes)."""
    return trace_payload_hash(rollout_traces_payload(result))


def compare_trace_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerances: dict[str, float],
    label: str = "repeat 1",
) -> tuple[list[str], dict[str, float]]:
    """Trace-by-trace comparison of two rollout trace payloads."""
    mismatches: list[str] = []
    max_diffs = {"requested": 0.0, "applied": 0.0, "final_angle_rad": 0.0, "force_n": 0.0}
    if candidate["n_ticks"] != reference["n_ticks"]:
        mismatches.append(
            f"{label}: n_ticks {candidate['n_ticks']} != {reference['n_ticks']}"
        )
        return mismatches, max_diffs  # lengths differ; elementwise is meaningless
    for key in ("first_success_tick", "termination_reason"):
        if candidate[key] != reference[key]:
            mismatches.append(f"{label}: {key} {candidate[key]!r} != {reference[key]!r}")
    if list(candidate["statuses"]) != list(reference["statuses"]):
        mismatches.append(f"{label}: adapter decision statuses differ")
    if list(candidate["contact"]) != list(reference["contact"]):
        mismatches.append(f"{label}: contact trace differs")
    if [f is None for f in candidate["force"]] != [f is None for f in reference["force"]]:
        mismatches.append(f"{label}: force trace availability differs")
    for key, tol_name in (("requested", "command_abs"), ("applied", "command_abs")):
        ref = np.asarray(reference[key], dtype=np.float64)
        cand = np.asarray(candidate[key], dtype=np.float64)
        diff = float(np.max(np.abs(cand - ref))) if ref.size else 0.0
        max_diffs[key] = max(max_diffs[key], diff)
        if diff > tolerances[tol_name]:
            mismatches.append(
                f"{label}: {key} command trace differs by {diff:.3g} "
                f"(> {tolerances[tol_name]:.3g})"
            )
    angle_diff = abs(candidate["final_angle_rad"] - reference["final_angle_rad"])
    max_diffs["final_angle_rad"] = angle_diff
    if angle_diff > tolerances["angle_abs_rad"]:
        mismatches.append(
            f"{label}: final angle differs by {angle_diff:.3g} rad "
            f"(> {tolerances['angle_abs_rad']:.3g})"
        )
    force_pairs = [
        (a, b)
        for a, b in zip(candidate["force"], reference["force"], strict=True)
        if a is not None and b is not None
    ]
    force_diff = max((abs(a - b) for a, b in force_pairs), default=0.0)
    max_diffs["force_n"] = force_diff
    if force_diff > tolerances["force_abs_n"]:
        mismatches.append(
            f"{label}: force trace differs by {force_diff:.3g} N "
            f"(> {tolerances['force_abs_n']:.3g})"
        )
    return mismatches, max_diffs


DETERMINISM_PROBE_KIND = "repeat_same_seed_fresh_process"
"""The determinism contract of this build: the k-th episode of a process is
bit-reproducible across fresh processes for the same seeds/configuration, but
same-seed repeats *within* one process are history-dependent (PhysX internal
state evolves per episode — measured at pose D4: 4 distinct trajectories in 6
in-process repeats, all exactly reproducible across processes). The probe
therefore compares the *first* fixed-seed rollout of the primary eval process
against the first rollout of one or more fresh replay processes with identical
reset seed, policy sampling seed, pose, checkpoint, and configuration."""


def determinism_probe_reference(result, *, seed: int) -> dict[str, Any]:
    """Pending fresh-process probe block for an eval payload (repeats=1).

    A replay invocation (``--determinism-replay``) re-runs the same rollout as
    the first episode of a fresh process and completes the block via
    :func:`determinism_probe_update`.
    """
    traces = rollout_traces_payload(result)
    return {
        "kind": DETERMINISM_PROBE_KIND,
        "seed": seed,
        "repeats": 1,
        "tolerances": dict(DETERMINISM_TOLERANCES),
        "trace_sha256": [trace_payload_hash(traces)],
        "reference_traces": traces,
        "max_abs_diffs": None,
        "mismatches": [],
        "passed": None,
        "note": "replay pending: rerun this eval with --determinism-replay",
    }


def determinism_probe_update(probe: dict[str, Any], result) -> dict[str, Any]:
    """Fold one fresh-process replay rollout into a pending/complete probe block."""
    if probe.get("kind") != DETERMINISM_PROBE_KIND:
        raise ValueError(f"unexpected determinism probe kind {probe.get('kind')!r}")
    candidate = rollout_traces_payload(result)
    label = f"repeat {probe['repeats']}"
    mismatches, max_diffs = compare_trace_payloads(
        probe["reference_traces"], candidate, probe["tolerances"], label=label
    )
    probe = dict(probe)
    probe["repeats"] = int(probe["repeats"]) + 1
    probe["trace_sha256"] = list(probe["trace_sha256"]) + [trace_payload_hash(candidate)]
    probe["mismatches"] = list(probe["mismatches"]) + mismatches
    previous = probe.get("max_abs_diffs") or {}
    probe["max_abs_diffs"] = {
        key: max(float(previous.get(key, 0.0)), value) for key, value in max_diffs.items()
    }
    probe["passed"] = not probe["mismatches"]
    probe.pop("note", None)
    return probe


def determinism_probe_report(
    results: list,
    *,
    seed: int,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """In-process repeat comparison over ``RolloutResult`` runs (pure helper).

    Used by unit tests to exercise the comparison machinery; production eval
    evidence uses the fresh-process probe (:data:`DETERMINISM_PROBE_KIND`)
    because same-seed repeats within one sim process are history-dependent.
    """
    if len(results) < 2:
        raise ValueError("determinism probe needs at least 2 repeat rollouts")
    tolerances = dict(DETERMINISM_TOLERANCES if tolerances is None else tolerances)
    reference = rollout_traces_payload(results[0])
    mismatches: list[str] = []
    max_diffs = {"requested": 0.0, "applied": 0.0, "final_angle_rad": 0.0, "force_n": 0.0}
    for index, result in enumerate(results[1:], start=1):
        repeat_mismatches, repeat_diffs = compare_trace_payloads(
            reference, rollout_traces_payload(result), tolerances, label=f"repeat {index}"
        )
        mismatches.extend(repeat_mismatches)
        for key, value in repeat_diffs.items():
            max_diffs[key] = max(max_diffs[key], value)
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
    "DETERMINISM_PROBE_KIND",
    "DETERMINISM_TOLERANCES",
    "aggregate_rollout_rows",
    "compare_trace_payloads",
    "contact_report",
    "determinism_probe_reference",
    "determinism_probe_report",
    "determinism_probe_update",
    "force_trace_evidence",
    "rollout_failure_label",
    "rollout_trace_hash",
    "rollout_traces_payload",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
    "trace_payload_hash",
]
