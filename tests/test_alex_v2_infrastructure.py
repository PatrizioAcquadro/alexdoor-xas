"""Pure acceptance tests for the additive Alex V2 artifact lineage."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import pytest

from alexdoor_xas import paths
from alexdoor_xas.assets import alex_v2 as alex_v2_loader
from alexdoor_xas.assets.alex_v2_contract import (
    DOOR_ACTUATOR_CONFIG_VERSION,
    DOOR_NON_RIGHT_ARM_DAMPING_SCALE,
    DOOR_RIGHT_ARM_ACTUATOR_NAME,
    DOOR_RIGHT_ARM_PD_GAINS,
    DOOR_RIGHT_ARM_PD_VERSION,
    EXPECTED_RUNTIME_JOINTS,
    EXPECTED_URDF_SHA256,
    AlexV2ContractError,
    RobotAssetRef,
    assert_checkpoint_runtime_compatible,
    derive_fixed_base_door_manifest,
    door_right_arm_pd_contract,
    validate_alex_v2_manifest,
)
from alexdoor_xas.assets.alex_v2_manifest import build_alex_v2_manifest
from alexdoor_xas.dataset.robot_asset import (
    dataset_robot_asset_payload,
    load_dataset_robot_asset,
)
from alexdoor_xas.recording import EpisodeMeta


def _manifest() -> dict:
    return build_alex_v2_manifest()


def _episode(manifest: dict, ref: RobotAssetRef):
    return SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id=ref.asset_id,
            robot_asset_sha256=ref.sha256,
        ),
        extras={"robot_asset_manifest": manifest},
    )


def test_manifest_freezes_all_29_runtime_joint_names_and_order() -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    assert len(EXPECTED_RUNTIME_JOINTS) == 29
    assert manifest["movable_joint_count"] == 29
    assert set(manifest["movable_joints"]) == set(EXPECTED_RUNTIME_JOINTS)
    assert ref.sha256 == EXPECTED_URDF_SHA256

    manifest["movable_joints"][5:7] = reversed(manifest["movable_joints"][5:7])
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(manifest)

    forged = _manifest()
    forged["urdf_sha256"] = "d" * 64
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged)


def test_manifest_rejects_nested_disagreement_and_rehashed_extra_inputs() -> None:
    forged_collision = _manifest()
    forged_collision["collision_profile"]["links"]["RIGHT_GRIPPER_Z_LINK"][0][
        "shape"
    ] = "sphere"
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_collision)

    forged_extra = _manifest()
    forged_extra["unexpected"] = "forged"
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_extra)

    forged_identity = _manifest()
    forged_identity["robot_asset_id"] = ""
    with pytest.raises(AlexV2ContractError, match="pinned URDF-derived manifest"):
        validate_alex_v2_manifest(forged_identity)

    forged_kind = _manifest()
    forged_kind["runtime_variant"] = None
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_kind)


def test_fixed_base_runtime_has_a_distinct_verified_identity() -> None:
    shared = _manifest()
    shared_ref = validate_alex_v2_manifest(shared)
    runtime = derive_fixed_base_door_manifest(shared)
    runtime_ref = validate_alex_v2_manifest(runtime)
    assert runtime_ref != shared_ref
    assert runtime["runtime_variant"]["fix_base"] is True
    assert runtime["runtime_variant"]["robot_tag"] == paths.ALEX_V2_ROBOT_TAG
    expected_pd = door_right_arm_pd_contract()
    assert runtime["runtime_variant"]["non_right_arm_damping_scale"] == 2.5
    assert runtime["fingerprint_inputs"]["non_right_arm_damping_scale"] == 2.5
    assert runtime["runtime_variant"]["right_arm_pd"] == expected_pd
    assert runtime["fingerprint_inputs"]["right_arm_pd"] == expected_pd
    assert expected_pd["version"] == DOOR_RIGHT_ARM_PD_VERSION
    assert expected_pd["actuator_name"] == DOOR_RIGHT_ARM_ACTUATOR_NAME
    assert [
        (item["joint_name"], item["stiffness"], item["damping"])
        for item in expected_pd["ordered_gains"]
    ] == list(DOOR_RIGHT_ARM_PD_GAINS)
    assert DOOR_NON_RIGHT_ARM_DAMPING_SCALE == 2.5
    assert DOOR_ACTUATOR_CONFIG_VERSION == "door-alex-v2-fixedbase-right-arm-pd-v2"
    assert DOOR_RIGHT_ARM_PD_VERSION == "door-alex-v2-right-arm-ik40-pd-v2"
    assert DOOR_RIGHT_ARM_PD_GAINS == (
        ("RIGHT_SHOULDER_Y", 600.0, 15.0),
        ("RIGHT_SHOULDER_X", 600.0, 15.0),
        ("RIGHT_SHOULDER_Z", 600.0, 15.0),
        ("RIGHT_ELBOW_Y", 600.0, 15.0),
        ("RIGHT_WRIST_Z", 150.0, 4.0),
        ("RIGHT_WRIST_X", 150.0, 4.0),
    )
    assert runtime_ref.manifest_fingerprint == (
        "b8d5672bd5f1f653640d8822c27b31409697efed53c01d57907f3a161acecc96"
    )
    assert runtime["actuator_config_version"] == DOOR_ACTUATOR_CONFIG_VERSION
    assert "actuator_config_version" not in shared

    forged = derive_fixed_base_door_manifest(shared)
    forged["runtime_variant"]["base_asset"]["manifest_fingerprint"] = "d" * 64
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged)

    forged_scale = derive_fixed_base_door_manifest(shared)
    forged_scale["runtime_variant"]["non_right_arm_damping_scale"] = 1.0
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_scale)

    forged_gain = derive_fixed_base_door_manifest(shared)
    forged_gain["runtime_variant"]["right_arm_pd"]["ordered_gains"][3][
        "damping"
    ] = 39.0
    with pytest.raises(AlexV2ContractError, match="canonical static-asset variant"):
        validate_alex_v2_manifest(forged_gain)


def _install_fake_v2_factory(monkeypatch, tmp_path, cfg):
    asset_path = tmp_path / "alex-v2.urdf"
    cfg.spawn.asset_path = str(asset_path)
    factory_calls = []

    def factory(path, *, fix_base, variant):
        factory_calls.append((path, fix_base, variant))
        return cfg

    asset = SimpleNamespace(urdf_path=asset_path)
    monkeypatch.setattr(
        alex_v2_loader,
        "build_alex_v2_door_asset",
        lambda **_kwargs: (asset, None),
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaaclab_assets.robots.alex":
            return SimpleNamespace(make_alex_v2_cfg=factory)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return factory_calls


def _fake_shared_arms_actuator():
    expressions = (
        ".*SHOULDER_Y",
        ".*SHOULDER_X",
        ".*SHOULDER_Z",
        ".*ELBOW_Y",
        ".*WRIST_Z",
        ".*WRIST_X",
        ".*GRIPPER_Z",
    )
    return SimpleNamespace(
        joint_names_expr=list(expressions),
        stiffness={expression: 5.0 + index for index, expression in enumerate(expressions)},
        damping={expression: 1.0 + index for index, expression in enumerate(expressions)},
        velocity_limit_sim={
            expression: 4.47 + index for index, expression in enumerate(expressions)
        },
        effort_limit_sim={
            expression: 20.86 + index for index, expression in enumerate(expressions)
        },
        armature={expression: 0.01 for expression in expressions},
        effort_limit=None,
        velocity_limit=None,
        friction=None,
        dynamic_friction=None,
        viscous_friction=None,
    )


def test_v2_loader_applies_production_damping_once_and_keeps_self_collision(
    monkeypatch, tmp_path
) -> None:
    cfg = SimpleNamespace(
        spawn=SimpleNamespace(
            asset_path="",
            self_collision=True,
            articulation_props=SimpleNamespace(enabled_self_collisions=True),
        ),
        actuators={
            "legs": SimpleNamespace(damping={".*HIP_X": 8.0, ".*KNEE_Y": 10.0}),
            "arms": _fake_shared_arms_actuator(),
        },
    )
    factory_calls = _install_fake_v2_factory(monkeypatch, tmp_path, cfg)

    loaded = alex_v2_loader.load_alex_v2_articulation_cfg()

    assert factory_calls == [(str(tmp_path / "alex-v2.urdf"), True, "standard")]
    assert loaded.spawn.self_collision is True
    assert loaded.spawn.articulation_props.enabled_self_collisions is True
    assert loaded.actuators["legs"].damping == {
        ".*HIP_X": 20.0,
        ".*KNEE_Y": 25.0,
    }
    assert loaded.actuators["arms"].joint_names_expr == [
        "LEFT_SHOULDER_Y",
        "LEFT_SHOULDER_X",
        "LEFT_SHOULDER_Z",
        "LEFT_ELBOW_Y",
        "LEFT_WRIST_Z",
        "LEFT_WRIST_X",
        "LEFT_GRIPPER_Z",
        "RIGHT_GRIPPER_Z",
    ]
    assert loaded.actuators["arms"].damping == {
        expression: (1.0 + index) * 2.5
        for index, expression in enumerate(
            (
                ".*SHOULDER_Y",
                ".*SHOULDER_X",
                ".*SHOULDER_Z",
                ".*ELBOW_Y",
                ".*WRIST_Z",
                ".*WRIST_X",
                ".*GRIPPER_Z",
            )
        )
    }
    right_arm = loaded.actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME]
    assert tuple(right_arm.joint_names_expr) == tuple(
        item[0] for item in DOOR_RIGHT_ARM_PD_GAINS
    )
    assert right_arm.stiffness == {
        joint_name: stiffness
        for joint_name, stiffness, _damping in DOOR_RIGHT_ARM_PD_GAINS
    }
    assert right_arm.damping == {
        joint_name: damping
        for joint_name, _stiffness, damping in DOOR_RIGHT_ARM_PD_GAINS
    }
    assert tuple(right_arm.velocity_limit_sim) == tuple(right_arm.joint_names_expr)
    assert tuple(right_arm.effort_limit_sim) == tuple(right_arm.joint_names_expr)
    assert tuple(right_arm.armature) == tuple(right_arm.joint_names_expr)


@pytest.mark.parametrize(
    ("spawn_self_collision", "root_self_collision", "damping", "error"),
    [
        (False, True, {".*HIP_X": 8.0}, "URDF self-collision enabled"),
        (True, False, {".*HIP_X": 8.0}, "articulation self-collision enabled"),
        (True, True, 8.0, "damping must be a non-empty mapping"),
    ],
)
def test_v2_loader_rejects_disabled_self_collision_or_scalar_damping(
    monkeypatch,
    tmp_path,
    spawn_self_collision,
    root_self_collision,
    damping,
    error,
) -> None:
    cfg = SimpleNamespace(
        spawn=SimpleNamespace(
            asset_path="",
            self_collision=spawn_self_collision,
            articulation_props=SimpleNamespace(
                enabled_self_collisions=root_self_collision
            ),
        ),
        actuators={
            "legs": SimpleNamespace(damping=damping),
            "arms": _fake_shared_arms_actuator(),
        },
    )
    _install_fake_v2_factory(monkeypatch, tmp_path, cfg)

    with pytest.raises((TypeError, ValueError), match=error):
        alex_v2_loader.load_alex_v2_articulation_cfg()


def test_v2_dataset_payload_embeds_and_revalidates_full_manifest(tmp_path) -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    payload = dataset_robot_asset_payload([_episode(manifest, ref), _episode(manifest, ref)])
    assert payload == {**ref.to_dict(), "manifest": manifest}

    (tmp_path / "meta.json").write_text(json.dumps({"robot_asset": payload}))
    loaded_ref, loaded_manifest = load_dataset_robot_asset(tmp_path, require=True)
    assert loaded_ref == ref
    assert loaded_manifest == manifest


def test_v2_dataset_rejects_missing_or_mixed_episode_provenance() -> None:
    manifest = _manifest()
    ref = validate_alex_v2_manifest(manifest)
    missing = SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    with pytest.raises(AlexV2ContractError, match="require robot asset provenance"):
        dataset_robot_asset_payload([missing])

    other = RobotAssetRef("other", "b" * 64)
    with pytest.raises(AlexV2ContractError, match="do not share one robot asset"):
        dataset_robot_asset_payload([_episode(manifest, ref), _episode(manifest, other)])


def test_dataset_payload_rejects_mixed_tasks_even_if_v2_is_not_first() -> None:
    proxy = SimpleNamespace(
        meta=SimpleNamespace(
            task="door_push",
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    v2 = SimpleNamespace(
        meta=SimpleNamespace(
            task=paths.ALEX_V2_TASK,
            robot_asset_id="",
            robot_asset_sha256="",
        ),
        extras={},
    )
    with pytest.raises(AlexV2ContractError, match="cannot mix episode tasks"):
        dataset_robot_asset_payload([proxy, v2])


def test_checkpoint_runtime_gate_fails_closed_and_labels_explicit_transfer() -> None:
    runtime = validate_alex_v2_manifest(_manifest())
    assert assert_checkpoint_runtime_compatible(runtime, runtime) == "v2_native"
    with pytest.raises(AlexV2ContractError, match="explicit cross-model evaluation flag"):
        assert_checkpoint_runtime_compatible(None, runtime)
    assert (
        assert_checkpoint_runtime_compatible(
            None, runtime, allow_cross_model_evaluation=True
        )
        == "v1_to_v2_transfer"
    )


def test_episode_meta_is_backward_compatible_but_can_carry_asset_identity() -> None:
    base = dict(
        episode_id="episode",
        task=paths.ALEX_V2_TASK,
        action_space="A2_ee_delta",
        robot=paths.ALEX_V2_ROBOT_TAG,
        scene="door",
        policy="scripted",
        seed=0,
        sim_dt=0.005,
        control_dt=0.02,
        chunk_len=1,
        created_utc="2026-01-01T00:00:00+00:00",
    )
    assert EpisodeMeta(**base).robot_asset_id == ""
    enriched = EpisodeMeta(**base, robot_asset_id="v2", robot_asset_sha256="c" * 64)
    assert enriched.to_dict()["robot_asset_sha256"] == "c" * 64
