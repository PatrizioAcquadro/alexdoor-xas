"""Pure import/config checks for the Phase 2 door-push DirectRLEnv."""

from __future__ import annotations

import importlib.util

import pytest


def test_door_push_env_package_imports_without_isaac_runtime() -> None:
    from alexdoor_xas.envs.door_task import DOOR_PUSH_ENV_ID

    assert DOOR_PUSH_ENV_ID == "AlexDoor-DoorPush-Proxy-v0"


def test_door_push_gym_registration_if_gymnasium_available() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this Python environment")

    import gymnasium as gym

    import alexdoor_xas.envs.door_task as door_task

    spec = gym.spec(door_task.DOOR_PUSH_ENV_ID)

    assert spec.entry_point == door_task.DOOR_PUSH_ENV_ENTRY_POINT
    assert spec.kwargs["env_cfg_entry_point"] == door_task.DOOR_PUSH_ENV_CFG_ENTRY_POINT
    assert spec.disable_env_checker is True


def test_door_push_env_cfg_contract_if_isaaclab_available() -> None:
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab is not installed in this Python environment")

    from alexdoor_xas.envs.door_task.door_env_cfg import (
        DOOR_TASK_ARTICULATION_PRIM_PATH,
        DOOR_TASK_SCENE_SOURCE_PRIM_PATH,
    )
    from alexdoor_xas.envs.door_task.door_push_env_cfg import (
        ACTION_TERMS,
        OBSERVATION_TERMS,
        PROXY_EE_PRIM_PATH,
        PROXY_EE_RADIUS_M,
        PROXY_EE_ROBOT_TAG,
        DoorPushEnvCfg,
    )

    cfg = DoorPushEnvCfg()

    assert len(ACTION_TERMS) == 6
    assert len(OBSERVATION_TERMS) == 9
    assert cfg.action_space == len(ACTION_TERMS)
    assert cfg.observation_space == len(OBSERVATION_TERMS)
    assert cfg.state_space == 0
    assert cfg.sim.device == "cpu"
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.episode_length_s == pytest.approx(10.0)
    assert cfg.scene.num_envs == 1
    assert cfg.door_task_scene.prim_path == DOOR_TASK_SCENE_SOURCE_PRIM_PATH
    assert cfg.door.prim_path == DOOR_TASK_ARTICULATION_PRIM_PATH
    assert cfg.door.spawn is None
    assert cfg.proxy_ee.prim_path == PROXY_EE_PRIM_PATH
    assert cfg.proxy_ee.spawn.radius == pytest.approx(PROXY_EE_RADIUS_M)
    # Velocity-driven dynamic sphere: not kinematic, gravity disabled.
    assert cfg.proxy_ee.spawn.rigid_props.kinematic_enabled is False
    assert cfg.proxy_ee.spawn.rigid_props.disable_gravity is True
    assert cfg.max_pos_delta_m == pytest.approx(0.02)
    assert cfg.max_rot_delta_rad == pytest.approx(0.05)
    assert PROXY_EE_ROBOT_TAG == "proxy_ee_sphere_v0"
