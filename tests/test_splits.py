"""Pure tests for the grouped, pose-stratified split contract (post-3.3 review).

No Isaac, no h5py: :func:`episode_content_key` and :func:`split_entries` are
duck-typed over loaded episode records, so synthetic records exercise the
exact production code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from alexdoor_xas.dataset import (
    SPLIT_NAMES,
    SplitEntry,
    assert_no_cross_split_duplicates,
    episode_content_key,
    load_split_payload,
    load_splits,
    make_grouped_splits,
    make_splits,
    save_splits,
    split_entries,
)


@dataclass
class _Record:
    """Duck-typed stand-in for ``EpisodeRecord`` (content fields only)."""

    episode_id: str
    t: np.ndarray
    actions: np.ndarray
    obs: dict[str, np.ndarray]
    success: bool = True
    final_door_angle: float = 0.9
    extras: dict[str, Any] = field(default_factory=dict)


def _record(episode_id: str, seed: int, pose: str | None = None, jitter: float = 0.0) -> _Record:
    rng = np.random.default_rng(seed)
    n = 20
    return _Record(
        episode_id=episode_id,
        t=np.arange(n, dtype=np.float64) / 60.0,
        actions=rng.standard_normal((n, 6)) + jitter,
        obs={
            "ee_pos_w": rng.standard_normal((n, 3)),
            "door_angle_rad": np.linspace(0.0, 0.9, n),
        },
        extras={"door_pose_id": pose} if pose else {},
    )


@dataclass
class _Dataset:
    records: list[_Record]


def _five_pose_entries(groups_per_pose: int = 9) -> list[SplitEntry]:
    """Synthetic v2_pose-shaped fixture: 5 poses x 10 episodes, fixed pair duplicated."""
    entries = []
    for p in range(5):
        pose = f"D{p}"
        # One duplicated fixed pair (2 episodes, 1 group) + 8 distinct episodes.
        entries.append(SplitEntry(f"{pose}-ep0", f"{pose}-group0", pose))
        entries.append(SplitEntry(f"{pose}-ep1", f"{pose}-group0", pose))
        for i in range(1, groups_per_pose):
            entries.append(SplitEntry(f"{pose}-ep{i + 1}", f"{pose}-group{i}", pose))
    return entries


# ── content keys ─────────────────────────────────────────────────────────────


def test_content_key_ignores_provenance_fields() -> None:
    a = _record("id-a", seed=3)
    b = _record("id-b", seed=3)  # same content, different episode id
    assert a.episode_id != b.episode_id
    assert episode_content_key(a) == episode_content_key(b)


def test_content_key_distinguishes_near_identical_trajectories() -> None:
    a = _record("id-a", seed=3)
    b = _record("id-b", seed=3)
    b.actions = a.actions.copy()
    b.actions[7, 2] += 1e-12  # near-but-not-identical must not collapse
    assert episode_content_key(a) != episode_content_key(b)


def test_content_key_sensitive_to_obs_and_outcome() -> None:
    base = _record("id-a", seed=3)
    changed_obs = _record("id-b", seed=3)
    changed_obs.obs["door_angle_rad"] = base.obs["door_angle_rad"] + 1e-9
    assert episode_content_key(base) != episode_content_key(changed_obs)
    changed_outcome = _record("id-c", seed=3)
    changed_outcome.success = False
    assert episode_content_key(base) != episode_content_key(changed_outcome)


def test_split_entries_reads_pose_and_defaults() -> None:
    dataset = _Dataset([_record("a", 0, pose="D1"), _record("b", 1)])
    entries = split_entries(dataset)
    assert entries[0].pose_id == "D1"
    assert entries[1].pose_id == "default"
    assert entries[0].episode_id == "a"


# ── grouped splits ───────────────────────────────────────────────────────────


def test_duplicates_never_cross_splits() -> None:
    entries = _five_pose_entries()
    for seed in range(10):
        splits, _ = make_grouped_splits(entries, seed=seed)
        assert_no_cross_split_duplicates(entries, splits)
        membership = {eid: name for name in SPLIT_NAMES for eid in splits[name]}
        for p in range(5):
            assert membership[f"D{p}-ep0"] == membership[f"D{p}-ep1"]


def test_every_pose_covered_in_val_and_test() -> None:
    splits, meta = make_grouped_splits(_five_pose_entries(), seed=0)
    for pose, info in meta["per_pose"].items():
        for name in ("val", "test"):
            assert info["episodes_per_split"][name] >= 1, (pose, name)
    # Disjoint + exhaustive over the 50-episode fixture.
    all_ids = [eid for name in SPLIT_NAMES for eid in splits[name]]
    assert len(all_ids) == len(set(all_ids)) == 50


def test_grouped_splits_deterministic_per_seed() -> None:
    entries = _five_pose_entries()
    assert make_grouped_splits(entries, seed=7) == make_grouped_splits(entries, seed=7)
    assert make_grouped_splits(entries, seed=7)[0] != make_grouped_splits(entries, seed=8)[0]


def test_grouped_splits_track_requested_sizes() -> None:
    splits, meta = make_grouped_splits(_five_pose_entries(), seed=0)
    assert meta["requested_sizes"] == {"train": 38, "val": 6, "test": 6}
    for name in SPLIT_NAMES:
        assert meta["actual_sizes"][name] == len(splits[name])
    # Whole groups only: sizes may deviate from the request, but the deviation
    # is recorded, bounded, and never negative for train.
    assert abs(meta["size_deviation"]["val"]) <= 2
    assert abs(meta["size_deviation"]["test"]) <= 2


def test_impossible_pose_coverage_fails_loudly() -> None:
    # One pose has only 2 independent groups: cannot place it in all 3 splits.
    entries = _five_pose_entries()
    entries += [
        SplitEntry("D5-ep0", "D5-group0", "D5"),
        SplitEntry("D5-ep1", "D5-group0", "D5"),
        SplitEntry("D5-ep2", "D5-group1", "D5"),
    ]
    with pytest.raises(ValueError, match="D5.*independent groups"):
        make_grouped_splits(entries, seed=0)


def test_group_spanning_poses_is_rejected() -> None:
    entries = [
        SplitEntry("a", "g0", "D0"),
        SplitEntry("b", "g0", "D1"),
        SplitEntry("c", "g1", "D0"),
    ]
    with pytest.raises(ValueError, match="spans door poses"):
        make_grouped_splits(entries, seed=0)


def test_metadata_audits_grouping() -> None:
    _, meta = make_grouped_splits(_five_pose_entries(), seed=5)
    assert meta["strategy"] == "grouped_pose_stratified"
    assert meta["grouping"] == "content_sha256"
    assert meta["seed"] == 5
    assert meta["n_episodes"] == 50
    assert meta["n_groups"] == 45  # 9 groups x 5 poses
    for pose in ("D0", "D1", "D2", "D3", "D4"):
        assert meta["per_pose"][pose]["n_groups"] == 9
        assert meta["per_pose"][pose]["n_episodes"] == 10
    group = meta["groups"]["D0-group0"]
    assert group["pose_id"] == "D0"
    assert group["episode_ids"] == ["D0-ep0", "D0-ep1"]
    assert group["split"] in SPLIT_NAMES


def test_assert_no_cross_split_duplicates_detects_leak() -> None:
    entries = [
        SplitEntry("a", "dup", "D0"),
        SplitEntry("b", "dup", "D0"),
        SplitEntry("c", "solo", "D0"),
    ]
    leaky = {"train": ["a"], "val": ["c"], "test": ["b"]}
    with pytest.raises(ValueError, match="cross split boundaries"):
        assert_no_cross_split_duplicates(entries, leaky)
    clean = {"train": ["a", "b"], "val": ["c"], "test": []}
    assert_no_cross_split_duplicates(entries, clean)


# ── id-only compatibility path + persistence ─────────────────────────────────


def test_make_splits_id_only_still_disjoint_exhaustive_deterministic() -> None:
    ids = [f"ep-{i}" for i in range(16)]
    splits = make_splits(ids, seed=3)
    assert make_splits(ids, seed=3) == splits
    all_ids = [eid for name in SPLIT_NAMES for eid in splits[name]]
    assert sorted(all_ids) == sorted(ids)
    assert len(splits["val"]) >= 1 and len(splits["test"]) >= 1
    with pytest.raises(ValueError):
        make_splits(ids[:2])


def test_save_splits_round_trips_metadata(tmp_path) -> None:
    entries = _five_pose_entries()
    splits, meta = make_grouped_splits(entries, seed=0)
    path = save_splits(tmp_path / "splits" / "v2_pose.json", splits, seed=0, metadata=meta)
    payload = load_split_payload(path)
    assert "split_fingerprint_sha256" not in payload
    assert payload["metadata"]["n_groups"] == meta["n_groups"]
    reloaded = load_splits(path, episode_ids=[e.episode_id for e in entries])
    assert reloaded == splits
