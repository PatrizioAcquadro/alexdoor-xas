"""Compatibility shim: dataset plumbing moved to ``policies.common.data``.

The ACT-named aliases are kept so existing imports, tests, and scripts stay
valid; new code should import from ``alexdoor_xas.policies.common.data``.
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

ActData = PolicyData
ActDataError = PolicyDataError
load_act_data = load_policy_data

__all__ = [
    "EPOCH_SEED_STRIDE",
    "ActData",
    "ActDataError",
    "load_act_data",
    "make_eval_factory",
    "make_train_factory",
    "normalize_batch",
]
