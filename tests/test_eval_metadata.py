"""Tests for checkpoint-owned evaluation metadata."""

from __future__ import annotations

from types import SimpleNamespace

from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.common.eval_metadata import checkpoint_metadata


def test_checkpoint_metadata_is_self_contained_and_dataset_independent() -> None:
    policy = SimpleNamespace(
        checkpoint_format="alexdoor_xas.act.v2",
        checkpoint_config={
            "dataset": {
                "task": "door_push_alex_v2",
                "space": "A2_ee_delta",
                "version": "v3_scale_master",
                "view_id": "v3_scale_n50",
                "obs_preset": "core_door_pose",
            }
        },
        action_space="A2_ee_delta",
        obs_preset="core_door_pose",
        model=SimpleNamespace(
            obs_dim=16,
            action_dim=6,
            cfg=ActModelCfg(chunk_size=8),
        ),
    )

    result = checkpoint_metadata(policy, "act")

    assert result["format"] == "alexdoor_xas.act.v2"
    assert result["policy"] == "act"
    assert result["dataset"]["view_id"] == "v3_scale_n50"
    assert result["observation_dim"] == 16
    assert result["action_dim"] == 6
    assert result["model_config"]["chunk_size"] == 8
    assert "dataset_provenance" not in result
