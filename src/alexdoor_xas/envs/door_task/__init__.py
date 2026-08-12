"""Gym registration for the supported Alex V2 door benchmark."""

from __future__ import annotations

DOOR_PUSH_ALEX_V2_ENV_ID = "AlexDoor-DoorPush-AlexV2-v0"
DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT = (
    "alexdoor_xas.envs.door_task.door_push_alex_v2_env:DoorPushAlexV2Env"
)
DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT = (
    "alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg:DoorPushAlexV2EnvCfg"
)


def _register_gym_envs() -> None:
    try:
        import gymnasium as gym
        from gymnasium.error import Error as GymError
    except ModuleNotFoundError:
        return

    registrations = (
        (
            DOOR_PUSH_ALEX_V2_ENV_ID,
            DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT,
            DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT,
        ),
    )
    for env_id, entry_point, cfg_entry_point in registrations:
        try:
            gym.spec(env_id)
        except GymError:
            gym.register(
                id=env_id,
                entry_point=entry_point,
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": cfg_entry_point},
            )


_register_gym_envs()

__all__ = [
    "DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT",
    "DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT",
    "DOOR_PUSH_ALEX_V2_ENV_ID",
]
