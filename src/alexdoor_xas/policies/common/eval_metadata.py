"""Compact checkpoint-owned metadata for closed-loop evaluation outputs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def checkpoint_metadata(policy: Any, policy_name: str) -> dict[str, Any]:
    """Describe the loaded policy without consulting its former training dataset."""

    dataset = dict((policy.checkpoint_config or {}).get("dataset") or {})
    return {
        "format": policy.checkpoint_format,
        "policy": policy_name,
        "dataset": dataset,
        "action_space": policy.action_space,
        "observation_preset": policy.obs_preset,
        "observation_dim": policy.model.obs_dim,
        "action_dim": policy.model.action_dim,
        "model_config": asdict(policy.model.cfg),
    }


__all__ = ["checkpoint_metadata"]
