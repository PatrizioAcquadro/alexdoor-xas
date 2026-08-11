"""Adapter-v1 (Phase 3.1): action representations -> executable robot commands.

The Adapter role of the system separation (guidelines §5): predicted actions
in any of the frozen spaces execute only through this layer, which validates,
corrects or rejects them against robot limits and door geometry, and logs
every decision. See ``knowledge/wiki/topics/action-representations-and-adapters.md``.

No Isaac imports — envs enter duck-typed via the frozen Phase 2 accessor
surface, so the layer is unit-testable and reusable by ACT / Diffusion Policy
/ VLA rollout evaluation alike.
"""

from .a2 import A2Adapter
from .a3 import A3Adapter, validate_object_frame
from .a4 import A4Adapter, A4AdapterCfg, A4ExecutionResult, StageResult
from .base import AdapterDecision, AdapterLog, AdapterStatus, AdapterWarning, StepContext
from .limits import (
    ALEX_V2_ROBOT_TAG,
    MAX_HINGE_ANGLE_RAD,
    PROXY_LIMITS,
    PROXY_ROBOT_TAG,
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
    "PROXY_LIMITS",
    "PROXY_ROBOT_TAG",
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
