"""Pure tests for rollout termination semantics (post-3.3 review WP: timing,
truncation, termination reasons, repeat-same-seed determinism helpers)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from alexdoor_xas.adapters import (
    PROXY_LIMITS,
    A2Adapter,
    AdapterDecision,
    AdapterLog,
    AdapterStatus,
    RolloutResult,
    rollout_chunks,
)
from alexdoor_xas.policies.common.rollout_eval import (
    determinism_probe_report,
    rollout_failure_label,
    rollout_trace_hash,
)
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv

SUCCESS_RAD = 0.3


def _push_source(horizon: int):
    """Constant -x push chunks of the given horizon (drives the fake door open)."""

    def source(ctx):
        del ctx
        return np.tile(np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), (horizon, 1))

    return source


def _run(env, horizon: int, **kwargs) -> RolloutResult:
    env.reset(seed=0)
    return rollout_chunks(
        env, _push_source(horizon), A2Adapter(PROXY_LIMITS), **kwargs
    )


class ScriptedAngleEnv(FakeDoorPushEnv):
    """Plays a prescribed hinge-angle sequence (index = executed ticks)."""

    def __init__(self, angles: list[float]):
        super().__init__()
        self._angles = list(angles)
        self._ticks = 0

    def reset(self, seed=None):
        self._ticks = 0
        return super().reset(seed)

    def step(self, action):
        result = super().step(action)
        self._ticks = min(self._ticks + 1, len(self._angles) - 1)
        return result

    def hinge_state(self):
        angle = self._angles[self._ticks]
        return (
            torch.tensor([angle], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        )


class TruncatingEnv(FakeDoorPushEnv):
    """Reports env truncation (and auto-resets, DirectRLEnv-style) at a fixed tick."""

    def __init__(self, truncate_at: int):
        super().__init__()
        self._truncate_at = truncate_at
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        return super().reset(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._steps += 1
        if self._steps >= self._truncate_at:
            steps = self._steps
            self.reset()  # post-reset state, exactly like DirectRLEnv inside step
            self._steps = steps
            truncated = torch.ones(1, dtype=torch.bool)
        return obs, reward, terminated, truncated, info


class SilentResetEnv(FakeDoorPushEnv):
    """Auto-resets its episode counter without reporting truncation (bug model)."""

    def __init__(self, reset_at: int):
        super().__init__()
        self._reset_at = reset_at
        self._steps = 0

    @property
    def episode_length_buf(self):
        return torch.tensor([self._steps])

    def reset(self, seed=None):
        self._steps = 0
        return super().reset(seed)

    def step(self, action):
        result = super().step(action)
        self._steps += 1
        if self._steps >= self._reset_at:
            self._steps = 0
        return result


# ── per-tick success timing ──────────────────────────────────────────────────


def test_first_success_is_exact_tick_for_h40_and_h8() -> None:
    # Reference crossing tick from single-step chunks (checked every tick by
    # construction), then H=40 (ACT-like) and H=8 (Diffusion Ta-like) chunks
    # must report the identical crossing tick and stop exactly there.
    reference = _run(FakeDoorPushEnv(), 1, success_angle_rad=SUCCESS_RAD)
    assert reference.termination_reason == "success"
    assert reference.success is True
    assert reference.first_success_tick == reference.n_ticks
    assert reference.final_angle_rad >= SUCCESS_RAD

    for horizon in (40, 8):
        result = _run(FakeDoorPushEnv(), horizon, success_angle_rad=SUCCESS_RAD)
        assert result.first_success_tick == reference.first_success_tick, horizon
        assert result.n_ticks == reference.first_success_tick, horizon
        assert result.termination_reason == "success"
        # No post-success execution: the crossing tick is the last executed one.
        assert len(result.decisions_per_tick) == result.n_ticks


def test_success_at_reset_state_executes_zero_ticks() -> None:
    env = ScriptedAngleEnv([SUCCESS_RAD + 0.1] * 5)
    env.reset(seed=0)
    result = rollout_chunks(
        env, _push_source(4), A2Adapter(PROXY_LIMITS), success_angle_rad=SUCCESS_RAD
    )
    assert result.n_ticks == 0
    assert result.first_success_tick == 0
    assert result.termination_reason == "success"


def test_cross_then_rebound_keeps_success_and_crossing_tick() -> None:
    angles = [0.0, 0.1, 0.2, 0.35, 0.28, 0.2, 0.1]
    env = ScriptedAngleEnv(angles)
    env.reset(seed=0)
    result = rollout_chunks(
        env,
        _push_source(4),
        A2Adapter(PROXY_LIMITS),
        max_ticks=6,
        success_angle_rad=SUCCESS_RAD,
        post_success_diagnostic=True,
    )
    assert result.success is True
    assert result.first_success_tick == 3  # angles[3] = 0.35 is the first crossing
    assert result.final_angle_rad < SUCCESS_RAD  # rebounded, but success is latched
    assert result.termination_reason == "tick_budget"  # diagnostic ran to budget

    # Without the diagnostic flag the rollout stops at the crossing tick.
    env = ScriptedAngleEnv(angles)
    env.reset(seed=0)
    stopped = rollout_chunks(
        env,
        _push_source(4),
        A2Adapter(PROXY_LIMITS),
        max_ticks=6,
        success_angle_rad=SUCCESS_RAD,
    )
    assert stopped.n_ticks == 3
    assert stopped.termination_reason == "success"
    assert stopped.final_angle_rad == pytest.approx(0.35)


# ── termination reasons ──────────────────────────────────────────────────────


def test_policy_exhaustion_and_tick_budget_reasons() -> None:
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    chunks = iter([np.zeros((2, 6)), None])
    result = rollout_chunks(env, lambda ctx: next(chunks), A2Adapter(PROXY_LIMITS))
    assert result.termination_reason == "policy_exhausted"
    assert result.success is None  # no threshold given

    budget = _run(FakeDoorPushEnv(), 4, max_ticks=6)
    assert budget.termination_reason == "tick_budget"
    assert budget.n_ticks == 6


def test_env_truncation_freezes_pre_reset_state() -> None:
    # Truncate one tick before the known success crossing: the door is open
    # (angle > 0) but the env auto-resets it to 0 inside the truncating step.
    reference = _run(FakeDoorPushEnv(), 1, success_angle_rad=SUCCESS_RAD)
    truncate_at = reference.first_success_tick - 1
    env = TruncatingEnv(truncate_at=truncate_at)
    env.reset(seed=0)
    result = rollout_chunks(
        env, _push_source(3), A2Adapter(PROXY_LIMITS), success_angle_rad=SUCCESS_RAD
    )
    assert result.termination_reason == "env_truncated"
    assert result.env_truncated is True
    assert result.n_ticks == truncate_at
    # Final state = last valid pre-step read, never the post-reset zeros.
    assert result.final_angle_rad > 0.0
    assert env.world.angle == 0.0  # env itself did reset
    # No post-reset per-tick capture for the truncating tick.
    assert len(result.contact_per_tick) == result.n_ticks - 1
    assert len(result.decisions_per_tick) == result.n_ticks
    assert result.success is False


def test_silent_mid_rollout_reset_fails_loudly() -> None:
    env = SilentResetEnv(reset_at=4)
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="auto-reset mid-rollout"):
        rollout_chunks(env, _push_source(3), A2Adapter(PROXY_LIMITS), max_ticks=9)


def test_failure_label_covers_new_reasons() -> None:
    base = dict(success=False, n_ticks=10, max_ticks=600, contact_ticks=3, n_rejected=0, notes="")
    assert rollout_failure_label(**base, termination_reason="env_truncated") == "env_truncated"
    assert (
        rollout_failure_label(**base, termination_reason="rejection_stop")
        == "stopped_on_rejection"
    )
    assert (
        rollout_failure_label(**{**base, "n_ticks": 600}, termination_reason="tick_budget")
        == "timeout_no_success"
    )
    assert (
        rollout_failure_label(**base, termination_reason="policy_exhausted")
        == "policy_stopped_early"
    )
    assert rollout_failure_label(**{**base, "success": True}) is None


def test_force_env_records_same_termination_fields() -> None:
    # ACT- and Diffusion-style rollouts share RolloutResult: the force-sensing
    # env exposes the identical timing/termination surface.
    result = _run(FakeForceDoorPushEnv(), 8, success_angle_rad=SUCCESS_RAD)
    assert result.termination_reason == "success"
    assert result.first_success_tick == result.n_ticks
    assert result.force_n_per_tick and result.force_n_per_tick[-1] is not None


# ── repeat-same-seed determinism helpers ─────────────────────────────────────


def _result_pair(perturb: bool) -> list[RolloutResult]:
    results = []
    for repeat in range(2):
        result = _run(FakeForceDoorPushEnv(), 8, success_angle_rad=SUCCESS_RAD)
        if perturb and repeat == 1:
            result.force_n_per_tick[-1] = (result.force_n_per_tick[-1] or 0.0) + 1.0
        results.append(result)
    return results


def test_determinism_probe_passes_on_identical_repeats() -> None:
    report = determinism_probe_report(_result_pair(perturb=False), seed=0)
    assert report["passed"] is True
    assert report["repeats"] == 2
    assert report["kind"] == "repeat_same_seed"
    assert len(set(report["trace_sha256"])) == 1
    assert report["mismatches"] == []


def test_determinism_probe_fails_beyond_tolerance() -> None:
    report = determinism_probe_report(_result_pair(perturb=True), seed=0)
    assert report["passed"] is False
    assert any("force trace" in m for m in report["mismatches"])
    assert len(set(report["trace_sha256"])) == 2


def test_determinism_probe_detects_tick_count_mismatch() -> None:
    a = _run(FakeDoorPushEnv(), 8, success_angle_rad=SUCCESS_RAD)
    b = _run(FakeDoorPushEnv(), 8, max_ticks=a.n_ticks - 2)
    report = determinism_probe_report([a, b], seed=0)
    assert report["passed"] is False
    assert any("n_ticks" in m for m in report["mismatches"])


def test_trace_hash_is_stable_and_content_sensitive() -> None:
    a, b = _result_pair(perturb=False)
    assert rollout_trace_hash(a) == rollout_trace_hash(b)
    log = AdapterLog()
    synthetic = RolloutResult(
        n_ticks=1,
        initial_angle_rad=0.0,
        final_angle_rad=0.1,
        log=log,
        decisions_per_tick=[
            AdapterDecision(
                status=AdapterStatus.ACCEPTED,
                requested=np.zeros(6),
                applied=np.zeros(6),
            )
        ],
        contact_per_tick=[None],
        force_n_per_tick=[None],
    )
    assert rollout_trace_hash(synthetic) != rollout_trace_hash(a)


def test_determinism_probe_requires_two_repeats() -> None:
    a = _run(FakeDoorPushEnv(), 8, success_angle_rad=SUCCESS_RAD)
    with pytest.raises(ValueError, match="at least 2"):
        determinism_probe_report([a], seed=0)


def test_success_angle_none_preserves_legacy_semantics() -> None:
    # Replay-style callers (adapter gate) pass no threshold: success stays
    # None, per-tick capture covers every executed tick, budget semantics hold.
    result = _run(FakeDoorPushEnv(), 5, max_ticks=15)
    assert result.success is None
    assert result.first_success_tick is None
    assert len(result.contact_per_tick) == result.n_ticks == 15
    assert math.isfinite(result.final_angle_rad)
