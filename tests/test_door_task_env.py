"""Pure import/config checks for the single-door DirectRLEnv shell."""

from __future__ import annotations

import importlib.util

import pytest


def test_door_task_env_package_imports_without_isaac_runtime() -> None:
    from alexdoor_xas.envs.door_task import DOOR_TASK_ENV_ID

    assert DOOR_TASK_ENV_ID == "AlexDoor-DoorTask-Direct-v0"


def test_door_task_gym_registration_if_gymnasium_available() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this Python environment")

    import gymnasium as gym

    import alexdoor_xas.envs.door_task as door_task

    spec = gym.spec(door_task.DOOR_TASK_ENV_ID)

    assert spec.entry_point == door_task.DOOR_TASK_ENV_ENTRY_POINT
    assert spec.kwargs["env_cfg_entry_point"] == door_task.DOOR_TASK_ENV_CFG_ENTRY_POINT
    assert spec.disable_env_checker is True


def test_door_task_env_cfg_contract_if_isaaclab_available() -> None:
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab is not installed in this Python environment")

    from alexdoor_xas.envs.door_task.door_env_cfg import (
        ACTION_TERMS,
        DOOR_TASK_ARTICULATION_PRIM_PATH,
        DOOR_TASK_SCENE_PRIM_PATH,
        DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
        OBSERVATION_TERMS,
        DoorTaskEnvCfg,
    )

    cfg = DoorTaskEnvCfg()

    assert ACTION_TERMS == ("noop_debug_action",)
    assert OBSERVATION_TERMS == ("door_angle_rad", "door_angular_velocity_rad_s")
    assert cfg.action_space == len(ACTION_TERMS)
    assert cfg.observation_space == len(OBSERVATION_TERMS)
    assert cfg.state_space == 0
    assert cfg.sim.device == "cpu"
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.scene.num_envs == 1
    assert cfg.scene.env_spacing == pytest.approx(3.0)
    assert cfg.scene.replicate_physics is True
    assert cfg.scene.clone_in_fabric is False
    assert cfg.door_task_scene.prim_path == DOOR_TASK_SCENE_SOURCE_PRIM_PATH
    assert cfg.door_task_scene.spawn.usd_path == ""
    assert cfg.door.prim_path == DOOR_TASK_ARTICULATION_PRIM_PATH
    assert cfg.door.spawn is None
    assert cfg.door.actuators == {}
    assert DOOR_TASK_SCENE_SOURCE_PRIM_PATH == "/World/envs/env_0/DoorTaskScene"
    assert DOOR_TASK_SCENE_PRIM_PATH == "/World/envs/env_.*/DoorTaskScene"
