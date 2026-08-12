"""Pure registration and configuration contracts for supported environments."""

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

SUPPORTED_ENVIRONMENTS = (
    (
        "AlexDoor-DoorTask-Direct-v0",
        "alexdoor_xas.envs.door_task.door_env:DoorTaskEnv",
        "alexdoor_xas.envs.door_task.door_env_cfg:DoorTaskEnvCfg",
    ),
    (
        "AlexDoor-DoorPush-Proxy-v0",
        "alexdoor_xas.envs.door_task.door_push_env:DoorPushEnv",
        "alexdoor_xas.envs.door_task.door_push_env_cfg:DoorPushEnvCfg",
    ),
    (
        "AlexDoor-DoorPush-AlexV2-v0",
        "alexdoor_xas.envs.door_task.door_push_alex_v2_env:DoorPushAlexV2Env",
        "alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg:DoorPushAlexV2EnvCfg",
    ),
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


def test_supported_environment_entry_points_are_explicit() -> None:
    actual = (
        (
            door_task.DOOR_TASK_ENV_ID,
            door_task.DOOR_TASK_ENV_ENTRY_POINT,
            door_task.DOOR_TASK_ENV_CFG_ENTRY_POINT,
        ),
        (
            door_task.DOOR_PUSH_ENV_ID,
            door_task.DOOR_PUSH_ENV_ENTRY_POINT,
            door_task.DOOR_PUSH_ENV_CFG_ENTRY_POINT,
        ),
        (
            door_task.DOOR_PUSH_ALEX_V2_ENV_ID,
            door_task.DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT,
            door_task.DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT,
        ),
    )
    assert actual == SUPPORTED_ENVIRONMENTS


def test_supported_environments_are_registered_if_gymnasium_available() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this Python environment")

    import gymnasium as gym

    for env_id, entry_point, cfg_entry_point in SUPPORTED_ENVIRONMENTS:
        spec = gym.spec(env_id)
        assert spec.entry_point == entry_point
        assert spec.kwargs["env_cfg_entry_point"] == cfg_entry_point
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
