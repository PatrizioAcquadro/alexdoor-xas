"""Validate, transform, and execute A2-A4 action representations."""

from .a2 import A2Adapter
from .a3 import A3Adapter, validate_object_frame
from .a4 import A4Adapter, A4AdapterCfg, A4ExecutionResult, StageResult, alex_v2_a4_cfg
from .base import AdapterDecision, AdapterLog, AdapterStatus, AdapterWarning, StepContext
from .limits import (
    ALEX_V2_ROBOT_TAG,
    MAX_HINGE_ANGLE_RAD,
    DoorPanelGeometry,
    RobotLimitsCfg,
    WorkspaceSphere,
    alex_v2_limits,
    limits_for_robot,
)
from .rollout import (
    TERMINATION_REASONS,
    ChunkSource,
    InvalidSimulatorStateError,
    RolloutResult,
    read_door_frame,
    read_joint_limits,
    read_step_context,
    replay_source,
    rollout_chunks,
    step_env,
)

__all__ = [
    "ALEX_V2_ROBOT_TAG",
    "MAX_HINGE_ANGLE_RAD",
    "TERMINATION_REASONS",
    "A2Adapter",
    "A3Adapter",
    "A4Adapter",
    "A4AdapterCfg",
    "A4ExecutionResult",
    "AdapterDecision",
    "AdapterLog",
    "AdapterStatus",
    "AdapterWarning",
    "ChunkSource",
    "InvalidSimulatorStateError",
    "DoorPanelGeometry",
    "RobotLimitsCfg",
    "RolloutResult",
    "StageResult",
    "StepContext",
    "WorkspaceSphere",
    "alex_v2_a4_cfg",
    "alex_v2_limits",
    "limits_for_robot",
    "validate_object_frame",
    "read_door_frame",
    "read_joint_limits",
    "read_step_context",
    "replay_source",
    "rollout_chunks",
    "step_env",
]
