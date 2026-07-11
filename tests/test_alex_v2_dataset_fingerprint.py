"""Regression tests for additive V2 provenance in dataset fingerprints."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from alexdoor_xas.dataset.loader import obs_matrix
from alexdoor_xas.dataset.normalize import dataset_fingerprint


def _dataset_without_asset_provenance():
    steps = 2
    record = SimpleNamespace(
        episode_id="episode_proxy",
        schema_version="alexdoor.episode.v1",
        n_steps=steps,
        meta={
            "seed": 7,
            "robot": "proxy_ee_sphere_v0",
            "scene": "combinedScene.usda",
            "policy": "scripted",
        },
        success=True,
        final_door_angle=0.42,
        failure_label=None,
        t=np.asarray([0.0, 1.0 / 60.0], dtype=np.float64),
        actions=np.arange(12, dtype=np.float64).reshape(steps, 6) / 100.0,
        obs={
            "ee_pos_w": np.zeros((steps, 3), dtype=np.float64),
            "ee_quat_w_xyzw": np.tile(
                np.asarray([0.0, 0.0, 0.0, 1.0]), (steps, 1)
            ),
            "door_angle_rad": np.asarray([0.0, 0.01], dtype=np.float64),
            "door_angular_velocity_rad_s": np.asarray(
                [0.0, 0.1], dtype=np.float64
            ),
        },
    )
    return SimpleNamespace(
        action_space="A2_ee_delta",
        task="door_push",
        meta={},
        records=[record],
    )


def _phase3_fingerprint_without_asset_provenance(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.action_space.encode())
    digest.update(dataset.task.encode())
    for record in sorted(dataset.records, key=lambda item: item.episode_id):
        digest.update(record.episode_id.encode())
        for key in ("seed", "robot", "scene", "policy"):
            digest.update(str(record.meta.get(key, "")).encode())
            digest.update(b"\0")
        digest.update(str(record.success).encode())
        digest.update(np.asarray([record.final_door_angle], dtype=np.float64).tobytes())
        digest.update(str(record.failure_label).encode())
        digest.update(np.asarray(record.t, dtype=np.float64).tobytes())
        digest.update(np.asarray(record.actions, dtype=np.float64).tobytes())
        digest.update(obs_matrix(record, "core").astype(np.float64).tobytes())
    return digest.hexdigest()


def test_provenance_free_fingerprint_preserves_the_exact_phase3_byte_stream() -> None:
    dataset = _dataset_without_asset_provenance()
    assert dataset_fingerprint(dataset) == _phase3_fingerprint_without_asset_provenance(dataset)


def test_v2_provenance_is_domain_separated_from_the_base_fingerprint() -> None:
    dataset = _dataset_without_asset_provenance()
    base = dataset_fingerprint(dataset)
    dataset.meta["robot_asset"] = {"id": "alex-v2", "sha256": "a" * 64}
    dataset.records[0].meta["robot_asset_id"] = "alex-v2"
    dataset.records[0].meta["robot_asset_sha256"] = "a" * 64
    assert dataset_fingerprint(dataset) != base
