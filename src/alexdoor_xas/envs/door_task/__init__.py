"""Gym registration for the Alex V2 door benchmark."""

from __future__ import annotations

import gymnasium as gym

DOOR_PUSH_ALEX_V2_ENV_ID = "AlexDoor-DoorPush-AlexV2-v0"

gym.register(
    id=DOOR_PUSH_ALEX_V2_ENV_ID,
    entry_point="alexdoor_xas.envs.door_task.door_push_alex_v2_env:DoorPushAlexV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg:DoorPushAlexV2EnvCfg"
        )
    },
)
