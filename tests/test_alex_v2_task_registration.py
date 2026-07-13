"""Pure registration contract for the fail-closed Alex V2 task."""

from alexdoor_xas.envs import door_task


def test_alex_v2_task_has_distinct_entry_points() -> None:
    assert door_task.DOOR_PUSH_ALEX_V2_ENV_ID == "AlexDoor-DoorPush-AlexV2-v0"
    assert "door_push_alex_v2_env:DoorPushAlexV2Env" in (
        door_task.DOOR_PUSH_ALEX_V2_ENV_ENTRY_POINT
    )
    assert "door_push_alex_v2_env_cfg:DoorPushAlexV2EnvCfg" in (
        door_task.DOOR_PUSH_ALEX_V2_ENV_CFG_ENTRY_POINT
    )
