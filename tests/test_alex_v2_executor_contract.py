"""Pure contract tests for V2 config injection and executor hooks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration
from alexdoor_xas.envs.door_task.alex_v2_runtime import (
    ALEX_V2_PRIM_PATH,
    AlexV2RuntimeContractError,
    inject_alex_v2_runtime_cfg,
)

_TASK_DIR = Path(__file__).parents[1] / "src" / "alexdoor_xas" / "envs" / "door_task"


class _FakeRobotCfg:
    def __init__(self) -> None:
        self.prim_path = "/World/Wrong"
        self.init_state = SimpleNamespace(pos=None, rot=None, joint_pos=None)

    def replace(self, *, prim_path: str):
        self.prim_path = prim_path
        return self


def _calibration() -> AlexV2DoorCalibration:
    payload = {
        "status": "validated",
        "base_pose": {
            "position_m": [-0.55, -0.25, 0.95],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        "ready_joint_pos": {
            "RIGHT_SHOULDER_Y": 0.2,
            "RIGHT_SHOULDER_X": -0.2,
            "RIGHT_SHOULDER_Z": 0.1,
            "RIGHT_ELBOW_Y": -0.8,
            "RIGHT_WRIST_Z": 0.05,
            "RIGHT_WRIST_X": 0.1,
        },
        # Synthetic test bounds only. Production values require a measured V2 arc.
        "reach_shell_m": [0.2, 0.8],
        "tool_frame": {
            "translation_m": [0.11, 0.0, -0.06],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "controller": {"contact_force_threshold_n": 2.5},
        "randomization_bounds": {},
    }
    return AlexV2DoorCalibration(Path("candidate.json"), payload)


def test_v2_runtime_injects_dedicated_asset_and_calibrated_init_state() -> None:
    env_cfg = SimpleNamespace(robot=None, contact_force_threshold_n=0.0)
    robot_cfg = _FakeRobotCfg()

    result = inject_alex_v2_runtime_cfg(env_cfg, robot_cfg, _calibration())

    assert result is env_cfg
    assert env_cfg.robot is robot_cfg
    assert robot_cfg.prim_path == ALEX_V2_PRIM_PATH
    assert robot_cfg.init_state.pos == (-0.55, -0.25, 0.95)
    assert robot_cfg.init_state.rot == (0.0, 0.0, 1.0, 0.0)
    assert env_cfg.contact_force_threshold_n == 2.5
    ready = robot_cfg.init_state.joint_pos
    assert len(ready) == 7
    assert ready["RIGHT_ELBOW_Y"] == -0.8
    catch_all = next(name for name in ready if name.startswith("(?!(?:"))
    assert ready[catch_all] == 0.0
    assert all(name in catch_all for name in _calibration().ready_joint_pos)


def test_v2_runtime_refuses_missing_dedicated_articulation_cfg() -> None:
    with pytest.raises(AlexV2RuntimeContractError, match="dedicated"):
        inject_alex_v2_runtime_cfg(SimpleNamespace(), None, _calibration())


def test_executor_source_hooks_offset_pose_point_jacobian_and_exact_gpu_force() -> None:
    source = (_TASK_DIR / "door_push_alex_v2_executor.py").read_text(encoding="utf-8")

    assert "class DoorPushAlexV2Executor(DoorPushRobotEnv)" in source
    assert "compose_offset_pose_xyzw(" in source
    assert "link_jacobian_to_point(" in source
    assert "def _ee_pose_w(" in source
    assert "def _solve_arm_ik(" in source
    assert "get_raw_contact_data" in source
    assert "sum_actor_contact_forces" in source
    assert "get_other_actor_paths_from_ids" in source
    assert "net_forces_w" not in source
    assert "def robot_asset_provenance(" in source
    assert "def alex_v2_calibration(" in source
    assert "def ee_contact_prim_path(" in source
    assert "def shoulder_position_world_m(" in source
    assert "def point_jacobian_w(" in source
    assert "point_jacobian = self.point_jacobian_w()" in source
    assert "load_alex_v2_articulation_cfg(fix_base=True)" in source
    assert source.index("load_alex_v2_articulation_cfg(fix_base=True)") < source.index(
        "super().__init__(cfg, render_mode, **kwargs)"
    )


def test_v2_cfg_requires_gpu_raw_contacts_without_physx_filter_patterns() -> None:
    source = (_TASK_DIR / "door_push_alex_v2_env_cfg.py").read_text(encoding="utf-8")
    base_cfg_source = (_TASK_DIR / "door_push_robot_env_cfg.py").read_text(encoding="utf-8")

    assert "filter_prim_paths_expr=[]" in source
    assert 'DOOR_PANEL_BODY_PRIM_PATH = f"{DOOR_TASK_ARTICULATION_PRIM_PATH}/Door"' in (
        base_cfg_source
    )
    assert "Cylinder_001" not in base_cfg_source
    assert "ContactSensorCfg(" in source
    assert "class DoorPushAlexV2EnvCfg(DoorPushRobotEnvCfg)" in source


def test_robot_base_has_no_asset_builder_or_robot_specific_ee_constants() -> None:
    env_source = (_TASK_DIR / "door_push_robot_env.py").read_text(encoding="utf-8")
    cfg_source = (_TASK_DIR / "door_push_robot_env_cfg.py").read_text(encoding="utf-8")

    assert "build_alex_articulation_cfg" not in env_source + cfg_source
    assert "ALEX_EE" not in env_source + cfg_source
    assert "requires an injected robot articulation config" in env_source


def test_candidate_env_is_explicitly_candidate_only_and_not_exported_or_registered() -> None:
    candidate_source = (_TASK_DIR / "door_push_alex_v2_calibration_env.py").read_text(
        encoding="utf-8"
    )
    registration_source = (_TASK_DIR / "__init__.py").read_text(encoding="utf-8")

    assert "candidate_only = True" in candidate_source
    assert "gym_registration_allowed = False" in candidate_source
    assert "__all__: list[str] = []" in candidate_source
    assert "door_push_alex_v2_calibration_env" not in registration_source
    assert "load_candidate_alex_v2_door_calibration" not in (
        _TASK_DIR / "door_push_alex_v2_env.py"
    ).read_text(encoding="utf-8")


def test_production_env_runs_shared_executor_after_full_validation() -> None:
    source = (_TASK_DIR / "door_push_alex_v2_env.py").read_text(encoding="utf-8")

    assert "class DoorPushAlexV2Env(DoorPushAlexV2Executor)" in source
    assert "load_alex_v2_door_calibration(" in source
    assert "super().__init__(" in source
    assert "executor has not passed" not in source
