"""Pure data-engine tests for Alex V2 episode asset provenance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import (
    AlexV2ContractError,
    derive_fixed_base_door_manifest,
    validate_alex_v2_manifest,
)
from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest
from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from conftest import FakeDoorPushEnv, make_test_engine_cfg


def _shared_manifest() -> dict[str, Any]:
    return build_alex_v2_manifest()


class _AssetEnv(FakeDoorPushEnv):
    def __init__(self, payload: Any):
        super().__init__()
        self.payload = payload
        self.reset_calls = 0

    def robot_asset_provenance(self) -> Any:
        return self.payload

    def reset(self, seed: int | None = None):
        self.reset_calls += 1
        return super().reset(seed)


def _run(env: FakeDoorPushEnv):
    return run_episode(
        env,
        plan_episodes(1, 0, 7)[0],
        DataEngineCfg(
            task=paths.ALEX_V2_TASK,
            robot=paths.ALEX_V2_ROBOT_TAG,
            limitations=("Synthetic provenance test.",),
            max_ticks=1,
        ),
    )


def test_env_without_asset_accessor_preserves_episode_shape() -> None:
    episode = run_episode(
        FakeDoorPushEnv(),
        plan_episodes(1, 0, 7)[0],
        make_test_engine_cfg(max_ticks=1),
    )

    assert episode.meta.robot_asset_id == ""
    assert episode.meta.robot_asset_sha256 == ""
    assert "robot_asset_manifest" not in episode.extras


def test_v2_episode_records_validated_identity_and_deep_copied_manifest() -> None:
    manifest = derive_fixed_base_door_manifest(_shared_manifest())
    ref = validate_alex_v2_manifest(manifest)
    payload = {**ref.to_dict(), "manifest": manifest}
    original = deepcopy(payload)
    env = _AssetEnv(payload)

    episode = _run(env)

    assert env.reset_calls == 1
    assert payload == original
    assert episode.meta.robot_asset_id == ref.asset_id
    assert episode.meta.robot_asset_sha256 == ref.sha256
    recorded = episode.extras["robot_asset_manifest"]
    assert recorded == manifest
    assert recorded is not manifest
    assert recorded["collision_profile"] is not manifest["collision_profile"]

    recorded["collision_profile"]["links"].clear()
    assert manifest["collision_profile"]["links"]


@pytest.mark.parametrize(
    "case",
    ("non_mapping", "missing_key", "non_mapping_manifest", "mismatched_reference"),
)
def test_malformed_v2_provenance_fails_before_reset(case: str) -> None:
    manifest = derive_fixed_base_door_manifest(_shared_manifest())
    ref = validate_alex_v2_manifest(manifest)
    payload: Any = {**ref.to_dict(), "manifest": manifest}
    if case == "non_mapping":
        payload = []
    elif case == "missing_key":
        del payload["manifest_fingerprint"]
    elif case == "non_mapping_manifest":
        payload["manifest"] = "not-a-manifest"
    else:
        payload["id"] = "forged-asset-id"
    env = _AssetEnv(payload)

    with pytest.raises(AlexV2ContractError):
        _run(env)

    assert env.reset_calls == 0


def test_forged_v2_manifest_fails_before_reset() -> None:
    manifest = derive_fixed_base_door_manifest(_shared_manifest())
    ref = validate_alex_v2_manifest(manifest)
    manifest["runtime_variant"]["non_right_arm_damping_scale"] = 1.0
    env = _AssetEnv({**ref.to_dict(), "manifest": manifest})

    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        _run(env)

    assert env.reset_calls == 0


def test_non_callable_v2_provenance_accessor_fails_before_reset() -> None:
    env = _AssetEnv(None)
    env.robot_asset_provenance = None  # type: ignore[method-assign]

    with pytest.raises(AlexV2ContractError, match="must be callable"):
        _run(env)

    assert env.reset_calls == 0
