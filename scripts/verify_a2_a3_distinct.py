#!/usr/bin/env python
"""A2/A3 distinguishability gate (no Kit): posed exports must differ exactly.

For every episode of a dataset version present in both the A2 (world EE delta)
and A3 (door-frame EE delta) exports:

- episodes at the default door orientation (yaw = 0) must be numerically
  identical across the two exports (the conversion is rotation-only — door
  *translation* must not leak into free-vector deltas);
- episodes at a yawed door pose must differ, and must satisfy the exact frozen
  conversion ``A3 = R_doorᵀ · A2`` per 3-vector (translation and rotation
  parts), within 1e-9.

This is the Problem-2 acceptance artifact of the local stabilization pass:
it fails on any dataset where A2 and A3 are accidentally identical under a
non-world-aligned door pose, and on any conversion drift.

Run through the official launcher (pure Python, no Kit)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_a2_a3_distinct.py --task door_push_alex_v2 --version v2_pose
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.action.frames import door_frame_from_body_pose
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.dataset import EpisodeDataset

YAW_IDENTITY_TOL_RAD = 1e-9
CONVERSION_TOL = 1e-9
MIN_DISTINCT_DELTA = 1e-4
"""A yawed episode whose A2/A3 exports differ by less than this is treated as
numerically identical — the failure this gate exists to catch."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=paths.DATASETS_DIR)
    parser.add_argument("--task", type=str, default=paths.ALEX_V2_TASK)
    parser.add_argument("--version", type=str, default="v2_pose")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.datasets_root / args.task
    a2 = EpisodeDataset(root / A2_EE_DELTA / args.version)
    a3 = EpisodeDataset(root / A3_OBJ_REL_EE_DELTA / args.version)
    if sorted(a2.episode_ids) != sorted(a3.episode_ids):
        print("FAIL: A2 and A3 exports do not share episode ids", flush=True)
        return 1

    per_pose: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"n": 0, "max_pair_diff": 0.0, "max_conversion_err": 0.0}
    )
    failures: list[str] = []
    for episode_id in a2.episode_ids:
        rec2 = a2.by_id(episode_id)
        rec3 = a3.by_id(episode_id)
        frame = door_frame_from_body_pose(
            np.asarray(rec2.extras["door_frame_pos_w"], dtype=np.float64),
            np.asarray(rec2.extras["door_frame_quat_w_xyzw"], dtype=np.float64),
        )
        yaw = float(np.arctan2(frame.rot[1, 0], frame.rot[0, 0]))
        pose_id = str(rec2.extras.get("door_pose_id"))
        label = f"episode {episode_id[:8]} (pose {pose_id}, seed {rec2.meta['seed']})"

        pair_diff = float(np.abs(rec2.actions - rec3.actions).max())
        expected_a3 = np.concatenate(
            [rec2.actions[:, :3] @ frame.rot, rec2.actions[:, 3:] @ frame.rot], axis=1
        )
        conversion_err = float(np.abs(rec3.actions - expected_a3).max())

        stats = per_pose[pose_id]
        stats["n"] += 1
        stats["max_pair_diff"] = max(stats["max_pair_diff"], pair_diff)
        stats["max_conversion_err"] = max(stats["max_conversion_err"], conversion_err)

        if conversion_err > CONVERSION_TOL:
            failures.append(f"{label}: conversion drift |A3 - R^T A2| = {conversion_err:.3e}")
        if abs(yaw) <= YAW_IDENTITY_TOL_RAD:
            if pair_diff > CONVERSION_TOL:
                failures.append(
                    f"{label}: yaw=0 but exports differ by {pair_diff:.3e} "
                    "(translation leaked into the delta conversion?)"
                )
        elif pair_diff < MIN_DISTINCT_DELTA:
            failures.append(
                f"{label}: yaw={yaw:+.4f} rad but A2 and A3 are numerically "
                f"identical (max diff {pair_diff:.3e}) — the pose variation is not "
                "reaching the export"
            )

    for pose_id in sorted(per_pose):
        stats = per_pose[pose_id]
        print(
            f"[pose {pose_id}] episodes={stats['n']} "
            f"max|A2-A3|={stats['max_pair_diff']:.6f} "
            f"max_conversion_err={stats['max_conversion_err']:.2e}",
            flush=True,
        )
    if failures:
        for failure in failures:
            print(f"[fail] {failure}", flush=True)
        print("FAIL: A2/A3 distinguishability gate failed.", flush=True)
        return 1
    print("PASS: A2/A3 exports are distinct under posed doors and exactly convertible.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
