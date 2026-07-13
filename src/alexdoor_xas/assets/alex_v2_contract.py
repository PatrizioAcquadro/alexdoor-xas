"""Pure-Python Alex V2 identity and downstream compatibility contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_manifest import (
    ALEX_V2_STANDARD_PROFILE,
    EXPECTED_ALEX_V2_URDF_SHA256,
    AlexV2ManifestError,
    build_alex_v2_manifest,
)

PROFILE = ALEX_V2_STANDARD_PROFILE
EXPECTED_URDF_SHA256 = EXPECTED_ALEX_V2_URDF_SHA256
DOOR_ACTUATOR_CONFIG_VERSION = "door-alex-v2-fixedbase-right-arm-pd-v2"
DOOR_NON_RIGHT_ARM_DAMPING_SCALE = 2.5
DOOR_RIGHT_ARM_PD_VERSION = "door-alex-v2-right-arm-ik40-pd-v2"
DOOR_RIGHT_ARM_ACTUATOR_NAME = "right_arm_project_pd"
DOOR_RIGHT_ARM_PD_GAINS = (
    ("RIGHT_SHOULDER_Y", 600.0, 15.0),
    ("RIGHT_SHOULDER_X", 600.0, 15.0),
    ("RIGHT_SHOULDER_Z", 600.0, 15.0),
    ("RIGHT_ELBOW_Y", 600.0, 15.0),
    ("RIGHT_WRIST_Z", 150.0, 4.0),
    ("RIGHT_WRIST_X", 150.0, 4.0),
)
DOOR_RUNTIME_MANIFEST_KIND = "alexdoor.alex_v2.fixedbase.v2"

# Frozen from a clean headless import of the standard static URDF. Isaac's
# order is not URDF document order; runtime consumers compare every name and
# position, never only the expected count of 29.
EXPECTED_RUNTIME_JOINTS = (
    "LEFT_HIP_X",
    "RIGHT_HIP_X",
    "SPINE_Z",
    "LEFT_HIP_Z",
    "RIGHT_HIP_Z",
    "NECK_Z",
    "LEFT_SHOULDER_Y",
    "RIGHT_SHOULDER_Y",
    "LEFT_HIP_Y",
    "RIGHT_HIP_Y",
    "NECK_Y",
    "LEFT_SHOULDER_X",
    "RIGHT_SHOULDER_X",
    "LEFT_KNEE_Y",
    "RIGHT_KNEE_Y",
    "LEFT_SHOULDER_Z",
    "RIGHT_SHOULDER_Z",
    "LEFT_ANKLE_Y",
    "RIGHT_ANKLE_Y",
    "LEFT_ELBOW_Y",
    "RIGHT_ELBOW_Y",
    "LEFT_ANKLE_X",
    "RIGHT_ANKLE_X",
    "LEFT_WRIST_Z",
    "RIGHT_WRIST_Z",
    "LEFT_WRIST_X",
    "RIGHT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "RIGHT_GRIPPER_Z",
)


class AlexV2ContractError(ValueError):
    """Raised when an Alex V2 artifact violates the frozen project contract."""


@dataclass(frozen=True)
class RobotAssetRef:
    """Minimal robot identity embedded in every downstream artifact."""

    asset_id: str
    sha256: str
    manifest_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise AlexV2ContractError("robot asset id must not be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise AlexV2ContractError("robot asset sha256 must be 64 lowercase hex characters")
        if self.manifest_fingerprint and (
            len(self.manifest_fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in self.manifest_fingerprint)
        ):
            raise AlexV2ContractError(
                "manifest fingerprint must be 64 lowercase hex characters"
            )

    def to_dict(self) -> dict[str, str]:
        payload = {"id": self.asset_id, "sha256": self.sha256}
        if self.manifest_fingerprint:
            payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotAssetRef:
        return cls(
            asset_id=str(value["id"]),
            sha256=str(value["sha256"]),
            manifest_fingerprint=str(value.get("manifest_fingerprint", "")),
        )


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest or fingerprint input using deterministic canonical JSON."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_alex_v2_manifest(manifest: Mapping[str, Any]) -> RobotAssetRef:
    """Validate a static or fixed-base Door manifest against the pinned URDF."""
    expected_static = _canonical_static_manifest()
    if "runtime_variant" not in manifest:
        if dict(manifest) != expected_static:
            raise AlexV2ContractError(
                "static Alex V2 manifest differs from the pinned URDF-derived manifest"
            )
        return _static_asset_ref(expected_static)

    expected_runtime = _derive_fixed_base_door_manifest(
        expected_static, _static_asset_ref(expected_static)
    )
    if dict(manifest) != expected_runtime:
        raise AlexV2ContractError(
            "Door runtime manifest differs from the exact canonical static-asset variant"
        )
    return RobotAssetRef(
        asset_id=str(expected_runtime["asset_id"]),
        sha256=str(expected_runtime["robot_asset_sha256"]),
        manifest_fingerprint=str(expected_runtime["fingerprint"]),
    )


def derive_fixed_base_door_manifest(static_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a distinct fixed-base Door runtime identity from the static asset."""
    static_ref = validate_alex_v2_manifest(static_manifest)
    if "runtime_variant" in static_manifest:
        raise AlexV2ContractError("expected a static asset manifest, got a runtime manifest")
    derived = _derive_fixed_base_door_manifest(static_manifest, static_ref)
    validate_alex_v2_manifest(derived)
    return derived


def _derive_fixed_base_door_manifest(
    static_manifest: Mapping[str, Any], static_ref: RobotAssetRef
) -> dict[str, Any]:
    derived = json.loads(json.dumps(dict(static_manifest), sort_keys=True))
    runtime_inputs = _door_fingerprint_inputs(static_ref)
    runtime_fingerprint = canonical_manifest_sha256(runtime_inputs)
    runtime_asset_id = f"{paths.ALEX_V2_ROBOT_TAG}:{runtime_fingerprint}"
    derived.update(
        {
            "asset_id": runtime_asset_id,
            "robot_asset_id": runtime_asset_id,
            "robot_asset_sha256": runtime_fingerprint,
            "fingerprint": runtime_fingerprint,
            "fingerprint_inputs": runtime_inputs,
            "runtime_variant": _door_runtime_variant(static_ref),
            "actuator_config_version": DOOR_ACTUATOR_CONFIG_VERSION,
        }
    )
    return derived


def _canonical_static_manifest() -> dict[str, Any]:
    try:
        return build_alex_v2_manifest()
    except AlexV2ManifestError as error:
        raise AlexV2ContractError(str(error)) from error


def _static_asset_ref(manifest: Mapping[str, Any]) -> RobotAssetRef:
    return RobotAssetRef(
        asset_id=str(manifest["asset_id"]),
        sha256=str(manifest["robot_asset_sha256"]),
        manifest_fingerprint=str(manifest["fingerprint"]),
    )


def _door_runtime_variant(base_ref: RobotAssetRef) -> dict[str, Any]:
    return {
        "kind": DOOR_RUNTIME_MANIFEST_KIND,
        "fix_base": True,
        "robot_tag": paths.ALEX_V2_ROBOT_TAG,
        "actuator_config_version": DOOR_ACTUATOR_CONFIG_VERSION,
        "non_right_arm_damping_scale": DOOR_NON_RIGHT_ARM_DAMPING_SCALE,
        "right_arm_pd": door_right_arm_pd_contract(),
        "base_asset": base_ref.to_dict(),
    }


def _door_fingerprint_inputs(base_ref: RobotAssetRef) -> dict[str, Any]:
    return {
        "kind": DOOR_RUNTIME_MANIFEST_KIND,
        "base_asset": base_ref.to_dict(),
        "robot_tag": paths.ALEX_V2_ROBOT_TAG,
        "fix_base": True,
        "merge_fixed_joints": True,
        "actuator_config_version": DOOR_ACTUATOR_CONFIG_VERSION,
        "non_right_arm_damping_scale": DOOR_NON_RIGHT_ARM_DAMPING_SCALE,
        "right_arm_pd": door_right_arm_pd_contract(),
        "ordered_runtime_joints": list(EXPECTED_RUNTIME_JOINTS),
    }


def door_right_arm_pd_contract() -> dict[str, Any]:
    """Return the exact ordered right-arm gains embedded in runtime identity."""
    return {
        "version": DOOR_RIGHT_ARM_PD_VERSION,
        "actuator_name": DOOR_RIGHT_ARM_ACTUATOR_NAME,
        "ordered_gains": [
            {
                "joint_name": joint_name,
                "stiffness": stiffness,
                "damping": damping,
            }
            for joint_name, stiffness, damping in DOOR_RIGHT_ARM_PD_GAINS
        ],
    }


def load_alex_v2_manifest(path: str | Path) -> tuple[dict[str, Any], RobotAssetRef]:
    """Read and validate an Alex V2 manifest."""
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AlexV2ContractError(
            f"cannot read Alex V2 manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AlexV2ContractError(f"Alex V2 manifest {manifest_path} must be a JSON object")
    return value, validate_alex_v2_manifest(value)


def assert_checkpoint_runtime_compatible(
    checkpoint_asset: RobotAssetRef | None,
    runtime_asset: RobotAssetRef,
    *,
    allow_cross_model_evaluation: bool = False,
) -> str:
    """Fail closed on asset mismatch and return the required result label.

    Legacy checkpoints have no asset reference and are treated as V1.  They may
    run against V2 only for an explicitly requested transfer evaluation; those
    results receive a label that cannot be mistaken for V2-trained performance.
    """
    if checkpoint_asset == runtime_asset:
        return "v2_native"
    if not allow_cross_model_evaluation:
        source = checkpoint_asset.asset_id if checkpoint_asset else "legacy_v1_unfingerprinted"
        raise AlexV2ContractError(
            f"checkpoint asset {source!r} is incompatible with runtime asset "
            f"{runtime_asset.asset_id!r}; pass the explicit cross-model evaluation flag "
            "only when measuring V1-to-V2 transfer"
        )
    return "v1_to_v2_transfer" if checkpoint_asset is None else "cross_model_transfer"


__all__ = [
    "DOOR_ACTUATOR_CONFIG_VERSION",
    "DOOR_NON_RIGHT_ARM_DAMPING_SCALE",
    "DOOR_RIGHT_ARM_ACTUATOR_NAME",
    "DOOR_RIGHT_ARM_PD_GAINS",
    "DOOR_RIGHT_ARM_PD_VERSION",
    "EXPECTED_RUNTIME_JOINTS",
    "EXPECTED_URDF_SHA256",
    "PROFILE",
    "AlexV2ContractError",
    "RobotAssetRef",
    "assert_checkpoint_runtime_compatible",
    "canonical_manifest_sha256",
    "derive_fixed_base_door_manifest",
    "door_right_arm_pd_contract",
    "load_alex_v2_manifest",
    "validate_alex_v2_manifest",
]
