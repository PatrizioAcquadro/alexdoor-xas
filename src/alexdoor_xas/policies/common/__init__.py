"""Policy-agnostic helpers shared by chunk policies (ACT, Diffusion, later VLA).

Everything here is generic over "a trained chunk policy" — the frozen obs
presets, the dataset/splits/norm-stats plumbing, open-loop inspection, and
closed-loop eval aggregation. Policy packages import from here; this package
never imports a policy, an adapter, or Isaac.
"""

from alexdoor_xas.policies.common.closed_loop import (
    aggregate_closed_loop,
    factual_rollout_row,
    protocol_rollouts,
)
from alexdoor_xas.policies.common.data import (
    EPOCH_SEED_STRIDE,
    PolicyData,
    PolicyDataError,
    load_policy_data,
    make_eval_factory,
    make_train_factory,
    normalize_batch,
)
from alexdoor_xas.policies.common.inspect import open_loop_report, predict_episode_open_loop
from alexdoor_xas.policies.common.obs import (
    OBS_CLIP,
    ROLLOUT_OBS_PRESETS,
    build_env_obs,
    stop_on_hinge_angle,
)

__all__ = [
    "EPOCH_SEED_STRIDE",
    "OBS_CLIP",
    "ROLLOUT_OBS_PRESETS",
    "PolicyData",
    "PolicyDataError",
    "aggregate_closed_loop",
    "build_env_obs",
    "factual_rollout_row",
    "load_policy_data",
    "make_eval_factory",
    "make_train_factory",
    "normalize_batch",
    "open_loop_report",
    "predict_episode_open_loop",
    "protocol_rollouts",
    "stop_on_hinge_angle",
]
