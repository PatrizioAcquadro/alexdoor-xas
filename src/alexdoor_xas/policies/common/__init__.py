"""Policy-agnostic helpers shared by chunk policies (ACT, Diffusion, later VLA).

Everything here is generic over "a trained chunk policy" — the frozen obs
presets, the dataset/splits/norm-stats plumbing, open-loop inspection, and
closed-loop eval aggregation. Policy packages import from here; this package
never imports a policy, an adapter, or Isaac.
"""

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
from alexdoor_xas.policies.common.rollout_eval import (
    aggregate_rollout_rows,
    scripted_reference_payload,
    seed_protocol,
    summarize_decision_warnings,
)

__all__ = [
    "EPOCH_SEED_STRIDE",
    "OBS_CLIP",
    "ROLLOUT_OBS_PRESETS",
    "PolicyData",
    "PolicyDataError",
    "aggregate_rollout_rows",
    "build_env_obs",
    "load_policy_data",
    "make_eval_factory",
    "make_train_factory",
    "normalize_batch",
    "open_loop_report",
    "predict_episode_open_loop",
    "scripted_reference_payload",
    "seed_protocol",
    "stop_on_hinge_angle",
    "summarize_decision_warnings",
]
