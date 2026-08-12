"""Pure registration and configuration contracts for the Alex V2 environment."""

from __future__ import annotations

import importlib.util

import pytest

from alexdoor_xas.envs import door_task
from alexdoor_xas.envs.door_task.door_contract import (
    DOOR_PUSH_ACTION_TERMS,
    DOOR_PUSH_OBSERVATION_TERMS,
    DOOR_TASK_ARTICULATION_PRIM_PATH,
    DOOR_TASK_SCENE_PRIM_PATH,
    DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
)
from alexdoor_xas.envs.door_task.door_runtime import resolve_hinge_joint_id

SUPPORTED_ENVIRONMENT = (
    "AlexDoor-DoorPush-AlexV2-v0",
    "alexdoor_xas.envs.door_task.door_push_alex_v2_env:DoorPushAlexV2Env",
    "alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg:DoorPushAlexV2EnvCfg",
)


def test_neutral_door_contract_is_complete() -> None:
    assert len(DOOR_PUSH_ACTION_TERMS) == 6
    assert len(DOOR_PUSH_OBSERVATION_TERMS) == 9
    assert DOOR_TASK_SCENE_SOURCE_PRIM_PATH == "/World/envs/env_0/DoorTaskScene"
    assert DOOR_TASK_SCENE_PRIM_PATH == "/World/envs/env_.*/DoorTaskScene"
    assert DOOR_TASK_ARTICULATION_PRIM_PATH.endswith("/DoorTaskScene/DoorTaskDoor")


@pytest.mark.parametrize(
    ("joint_names", "expected"),
    [
        (["Hinge"], 0),
        (["shoulder", "Door/Hinge"], 1),
        (["the_only_joint"], 0),
    ],
)
def test_resolve_hinge_joint_id(joint_names: list[str], expected: int) -> None:
    assert resolve_hinge_joint_id(joint_names, "Hinge") == expected


def test_resolve_hinge_joint_id_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="could not identify one hinge joint"):
        resolve_hinge_joint_id(["Hinge", "Door/Hinge"], "Hinge")


def test_supported_environment_entry_point_is_explicit() -> None:
    actual = (
        door_task.DOOR_PUSH_ALEX_V2_ENV_ID,
        door_task.DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT,
        door_task.DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT,
    )
    assert actual == SUPPORTED_ENVIRONMENT


def test_supported_environment_is_registered_if_gymnasium_available() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this Python environment")

    import gymnasium as gym

    env_id, entry_point, cfg_entry_point = SUPPORTED_ENVIRONMENT
    spec = gym.spec(env_id)
    assert spec.entry_point == entry_point
    assert spec.kwargs["env_cfg_entry_point"] == cfg_entry_point
    assert spec.disable_env_checker is True


def test_alex_v2_env_cfg_contract_if_isaaclab_available() -> None:
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab is not installed in this Python environment")

    from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (
        ALEX_V2_ARM_JOINT_NAMES,
        ALEX_V2_EE_BODY_NAME,
        DoorPushAlexV2EnvCfg,
    )

    cfg = DoorPushAlexV2EnvCfg()

    assert cfg.action_space == len(DOOR_PUSH_ACTION_TERMS)
    assert cfg.observation_space == len(DOOR_PUSH_OBSERVATION_TERMS)
    assert cfg.state_space == 0
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.scene.num_envs == 1
    assert cfg.door_task_scene.prim_path == DOOR_TASK_SCENE_SOURCE_PRIM_PATH
    assert cfg.door.prim_path == DOOR_TASK_ARTICULATION_PRIM_PATH
    assert cfg.ee_body_name == ALEX_V2_EE_BODY_NAME
    assert tuple(cfg.arm_joint_names) == ALEX_V2_ARM_JOINT_NAMES
