"""Compatibility shim: helpers moved to ``policies.common.rollout_eval``."""

from alexdoor_xas.policies.common.rollout_eval import (
    DETERMINISM_PROBE_KIND,
    aggregate_rollout_rows,
    compare_trace_payloads,
    contact_report,
    determinism_probe_reference,
    determinism_probe_report,
    determinism_probe_update,
    final_ee_state,
    force_trace_evidence,
    rollout_trace_hash,
    rollout_traces_payload,
    scripted_reference_payload,
    seed_protocol,
    summarize_decision_warnings,
    trace_payload_hash,
)

__all__ = [
    "DETERMINISM_PROBE_KIND",
    "aggregate_rollout_rows",
    "compare_trace_payloads",
    "contact_report",
    "determinism_probe_reference",
    "determinism_probe_report",
    "determinism_probe_update",
    "final_ee_state",
    "force_trace_evidence",
    "rollout_trace_hash",
    "rollout_traces_payload",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
    "trace_payload_hash",
]
