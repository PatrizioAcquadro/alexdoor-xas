"""Scripted controller preset sourced only from validated Alex V2 calibration."""

from __future__ import annotations

from dataclasses import replace

from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration
from alexdoor_xas.policies.scripted.door_push import DoorPushControllerCfg, VariationBounds


def alex_v2_push_cfg(calibration: AlexV2DoorCalibration) -> DoorPushControllerCfg:
    values = calibration.controller
    return replace(
        DoorPushControllerCfg(),
        push_radius_frac=float(values["push_radius_frac"]),
        push_height_m=float(values["push_height_m"]),
        approach_standoff_m=float(values["approach_standoff_m"]),
        align_standoff_m=float(values["align_standoff_m"]),
        pre_contact_clearance_m=float(values["pre_contact_clearance_m"]),
        contact_clearance_m=float(values["contact_clearance_m"]),
        contact_approach_max_step_m=float(values["contact_approach_max_step_m"]),
        release_standoff_m=float(values["release_standoff_m"]),
    )


def alex_v2_variation_bounds(calibration: AlexV2DoorCalibration) -> VariationBounds:
    values = calibration.randomization_bounds
    return VariationBounds(
        start_offset_low=tuple(float(item) for item in values["start_offset_low"]),
        start_offset_high=tuple(float(item) for item in values["start_offset_high"]),
        push_radius_frac_range=tuple(float(item) for item in values["push_radius_frac_range"]),
        push_height_m_range=tuple(float(item) for item in values["push_height_m_range"]),
    )


__all__ = ["alex_v2_push_cfg", "alex_v2_variation_bounds"]
