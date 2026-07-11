"""Pure tests for candidate-only Alex V2 right-arm PD splitting."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from alexdoor_xas.assets.alex_v2_candidate_pd import (
    PRODUCTION_RIGHT_ARM_PD_PROFILE,
    RIGHT_ARM_PD_JOINTS,
    RIGHT_ARM_PD_PROFILES,
    apply_candidate_right_arm_pd,
    apply_production_right_arm_pd,
    apply_right_arm_pd_profile_selection,
    candidate_right_arm_pd_profile_names,
)
from alexdoor_xas.assets.alex_v2_contract import DOOR_RIGHT_ARM_ACTUATOR_NAME

_ARM_EXPRESSIONS = (
    ".*SHOULDER_Y",
    ".*SHOULDER_X",
    ".*SHOULDER_Z",
    ".*ELBOW_Y",
    ".*WRIST_Z",
    ".*WRIST_X",
    ".*GRIPPER_Z",
)


def _arms_actuator():
    return SimpleNamespace(
        joint_names_expr=list(_ARM_EXPRESSIONS),
        stiffness={
            ".*SHOULDER_Y": 26.78,
            ".*SHOULDER_X": 26.78,
            ".*SHOULDER_Z": 23.76,
            ".*ELBOW_Y": 23.76,
            ".*WRIST_Z": 5.0,
            ".*WRIST_X": 5.0,
            ".*GRIPPER_Z": 2.0,
        },
        damping={
            ".*SHOULDER_Y": 20.0,
            ".*SHOULDER_X": 20.0,
            ".*SHOULDER_Z": 10.0,
            ".*ELBOW_Y": 10.0,
            ".*WRIST_Z": 2.5,
            ".*WRIST_X": 2.5,
            ".*GRIPPER_Z": 1.25,
        },
        velocity_limit_sim={
            expression: 4.47 + index
            for index, expression in enumerate(_ARM_EXPRESSIONS)
        },
        effort_limit_sim={
            expression: 20.86 + index
            for index, expression in enumerate(_ARM_EXPRESSIONS)
        },
        armature={expression: 0.01 for expression in _ARM_EXPRESSIONS},
        friction={expression: 0.0 for expression in _ARM_EXPRESSIONS},
    )


def _cfg():
    return SimpleNamespace(
        spawn=SimpleNamespace(
            self_collision=True,
            articulation_props=SimpleNamespace(enabled_self_collisions=True),
        ),
        actuators={
            "legs": SimpleNamespace(damping={".*HIP_X": 20.0875}),
            "torso_head": SimpleNamespace(damping={"SPINE_Z": 20.0875}),
            "arms": _arms_actuator(),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def test_profile_names_and_gain_contract_are_deterministic() -> None:
    assert candidate_right_arm_pd_profile_names() == (
        "stable_4x_v1",
        "balanced_3_5x_v1",
        "responsive_k125_v1",
    )
    for profile in RIGHT_ARM_PD_PROFILES.values():
        assert tuple(profile) == RIGHT_ARM_PD_JOINTS
        assert all(set(gains) == {"stiffness", "damping"} for gains in profile.values())
        assert all(
            gains["stiffness"] > 0.0 and gains["damping"] > 0.0
            for gains in profile.values()
        )


def test_production_splits_exact_right_arm_and_preserves_every_other_actuator() -> None:
    cfg = _cfg()
    legs_before = deepcopy(cfg.actuators["legs"])
    torso_before = deepcopy(cfg.actuators["torso_head"])
    arms_before = deepcopy(cfg.actuators["arms"])

    evidence = apply_production_right_arm_pd(cfg)

    assert cfg.actuators["legs"].__dict__ == legs_before.__dict__
    assert cfg.actuators["torso_head"].__dict__ == torso_before.__dict__
    assert cfg.spawn.self_collision is True
    assert cfg.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.soft_joint_pos_limit_factor == 1.0

    retained = cfg.actuators["arms"]
    right = cfg.actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME]
    assert tuple(right.joint_names_expr) == RIGHT_ARM_PD_JOINTS
    assert set(retained.joint_names_expr).isdisjoint(RIGHT_ARM_PD_JOINTS)
    assert set(retained.joint_names_expr) == {
        "LEFT_SHOULDER_Y",
        "LEFT_SHOULDER_X",
        "LEFT_SHOULDER_Z",
        "LEFT_ELBOW_Y",
        "LEFT_WRIST_Z",
        "LEFT_WRIST_X",
        "LEFT_GRIPPER_Z",
        "RIGHT_GRIPPER_Z",
    }
    assert retained.stiffness == arms_before.stiffness
    assert retained.damping == arms_before.damping
    assert retained.velocity_limit_sim == arms_before.velocity_limit_sim
    assert retained.effort_limit_sim == arms_before.effort_limit_sim
    assert retained.armature == arms_before.armature
    assert retained.friction == arms_before.friction

    expected = RIGHT_ARM_PD_PROFILES["stable_4x_v1"]
    assert right.stiffness == {
        name: expected[name]["stiffness"] for name in RIGHT_ARM_PD_JOINTS
    }
    assert right.damping == {
        name: expected[name]["damping"] for name in RIGHT_ARM_PD_JOINTS
    }
    for name, source_expression in zip(
        RIGHT_ARM_PD_JOINTS, _ARM_EXPRESSIONS[:-1], strict=True
    ):
        assert (
            right.velocity_limit_sim[name]
            == arms_before.velocity_limit_sim[source_expression]
        )
        assert (
            right.effort_limit_sim[name]
            == arms_before.effort_limit_sim[source_expression]
        )
        assert right.armature[name] == arms_before.armature[source_expression]
        assert right.friction[name] == arms_before.friction[source_expression]
        assert evidence["gains"][name] == {
            **expected[name],
            "velocity_limit_sim": arms_before.velocity_limit_sim[source_expression],
            "effort_limit_sim": arms_before.effort_limit_sim[source_expression],
            "armature": arms_before.armature[source_expression],
        }
    assert evidence["right_arm_only"] is True
    assert evidence["position_limits"] == {"source": "URDF", "modified": False}
    assert evidence["scope"] == "production_door_v2"


def test_candidate_overrides_only_the_isolated_production_right_arm() -> None:
    cfg = _cfg()
    apply_production_right_arm_pd(cfg)
    non_right_before = {
        name: deepcopy(actuator)
        for name, actuator in cfg.actuators.items()
        if name != DOOR_RIGHT_ARM_ACTUATOR_NAME
    }
    production_right = deepcopy(cfg.actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME])

    evidence = apply_candidate_right_arm_pd(
        cfg, profile_name="balanced_3_5x_v1"
    )

    for name, actuator in non_right_before.items():
        assert cfg.actuators[name].__dict__ == actuator.__dict__
    candidate = cfg.actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME]
    assert candidate.stiffness == {
        name: RIGHT_ARM_PD_PROFILES["balanced_3_5x_v1"][name]["stiffness"]
        for name in RIGHT_ARM_PD_JOINTS
    }
    assert candidate.damping == {
        name: RIGHT_ARM_PD_PROFILES["balanced_3_5x_v1"][name]["damping"]
        for name in RIGHT_ARM_PD_JOINTS
    }
    assert candidate.velocity_limit_sim == production_right.velocity_limit_sim
    assert candidate.effort_limit_sim == production_right.effort_limit_sim
    assert candidate.armature == production_right.armature
    assert evidence["overrides_production_right_arm"] is True
    assert evidence["production_config_modified"] is False
    assert evidence["production_manifest_modified"] is False


def test_production_v2_profile_selects_ik40_without_candidate_override() -> None:
    cfg = _cfg()
    apply_production_right_arm_pd(cfg)
    actuators_before = deepcopy(cfg.actuators)

    selection = apply_right_arm_pd_profile_selection(
        cfg, profile_name="production_right_arm_pd_v2"
    )

    assert PRODUCTION_RIGHT_ARM_PD_PROFILE == "production_right_arm_pd_v2"
    assert selection == {
        "requested_profile": "production_right_arm_pd_v2",
        "effective_profile": "production_right_arm_pd_v2",
        "uses_production_right_arm_pd": True,
        "candidate_override": None,
    }
    assert cfg.actuators.keys() == actuators_before.keys()
    for name, actuator in cfg.actuators.items():
        assert actuator.__dict__ == actuators_before[name].__dict__


def test_removed_production_v1_spelling_is_rejected() -> None:
    cfg = _cfg()
    apply_production_right_arm_pd(cfg)
    actuators_before = deepcopy(cfg.actuators)

    with pytest.raises(ValueError, match="unknown right-arm PD candidate"):
        apply_right_arm_pd_profile_selection(
            cfg, profile_name="production_right_arm_pd_" + "v1"
        )

    for name, actuator in cfg.actuators.items():
        assert actuator.__dict__ == actuators_before[name].__dict__


def test_candidate_profile_selection_remains_an_explicit_override() -> None:
    cfg = _cfg()
    apply_production_right_arm_pd(cfg)

    selection = apply_right_arm_pd_profile_selection(
        cfg, profile_name="balanced_3_5x_v1"
    )

    assert selection["effective_profile"] == "balanced_3_5x_v1"
    assert selection["uses_production_right_arm_pd"] is False
    assert selection["candidate_override"]["overrides_production_right_arm"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cfg: setattr(cfg.spawn, "self_collision", False),
            "URDF self-collision enabled",
        ),
        (
            lambda cfg: setattr(
                cfg.spawn.articulation_props, "enabled_self_collisions", False
            ),
            "articulation self-collision enabled",
        ),
        (
            lambda cfg: cfg.actuators["arms"].joint_names_expr.reverse(),
            "joint expressions differ",
        ),
        (
            lambda cfg: cfg.actuators["arms"].velocity_limit_sim.pop(".*ELBOW_Y"),
            "missing source expression",
        ),
    ],
)
def test_production_rejects_unsafe_or_malformed_source_without_mutating_actuators(
    mutation, match: str
) -> None:
    cfg = _cfg()
    mutation(cfg)
    actuators_before = deepcopy(cfg.actuators)

    with pytest.raises((TypeError, ValueError), match=match):
        apply_production_right_arm_pd(cfg)

    assert cfg.actuators.keys() == actuators_before.keys()
    assert DOOR_RIGHT_ARM_ACTUATOR_NAME not in cfg.actuators


def test_candidate_rejects_unknown_profile_and_supports_future_reoverride() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="unknown right-arm PD candidate"):
        apply_candidate_right_arm_pd(cfg, profile_name="not-a-profile")

    apply_production_right_arm_pd(cfg)
    apply_candidate_right_arm_pd(cfg, profile_name="balanced_3_5x_v1")
    apply_candidate_right_arm_pd(cfg, profile_name="responsive_k125_v1")
    assert cfg.actuators[DOOR_RIGHT_ARM_ACTUATOR_NAME].damping == {
        name: RIGHT_ARM_PD_PROFILES["responsive_k125_v1"][name]["damping"]
        for name in RIGHT_ARM_PD_JOINTS
    }
