"""Compatibility shim: helpers moved to ``policies.common.rollout_eval``."""

from alexdoor_xas.policies.common.rollout_eval import (
    aggregate_rollout_rows,
    contact_report,
    rollout_failure_label,
    scripted_reference_payload,
    seed_protocol,
    summarize_decision_warnings,
)

__all__ = [
    "aggregate_rollout_rows",
    "contact_report",
    "rollout_failure_label",
    "scripted_reference_payload",
    "seed_protocol",
    "summarize_decision_warnings",
]
