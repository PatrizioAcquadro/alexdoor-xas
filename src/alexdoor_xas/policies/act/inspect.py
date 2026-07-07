"""Compatibility shim: open-loop inspection moved to ``policies.common.inspect``."""

from alexdoor_xas.policies.common.inspect import open_loop_report, predict_episode_open_loop

__all__ = ["open_loop_report", "predict_episode_open_loop"]
