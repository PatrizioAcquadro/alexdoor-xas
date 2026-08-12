"""Per-run Phase 2 report/manifest (markdown) summarizing one baseline run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from alexdoor_xas.recording import EpisodeBuffer


def write_run_report(
    path: str | Path,
    *,
    episodes: list[EpisodeBuffer],
    per_episode_metrics: list[dict[str, Any]],
    aggregate: dict[str, Any],
    exports: dict[str, Path] | None,
    plots: dict[str, Path] | None,
    videos: dict[str, Any] | None,
    limitations: list[str],
) -> Path:
    """Write the run manifest; returns the report path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = episodes[0].meta if episodes else None

    lines: list[str] = ["# Scripted door-push baseline — run report", ""]
    if meta is not None:
        lines += [
            "## Setup",
            "",
            f"- task: `{meta.task}`  scene: `{meta.scene}`  robot: `{meta.robot}`",
            f"- policy: `{meta.policy}`  recorded action space: `{meta.action_space}`",
            f"- sim_dt: {meta.sim_dt:.6g} s  control_dt: {meta.control_dt:.6g} s",
            f"- controller cfg: `{json.dumps(episodes[0].extras.get('controller_cfg'))}`",
            f"- engine cfg: `{json.dumps(episodes[0].extras.get('engine_cfg'))}`",
            "",
        ]

    # Force columns only when the run recorded force-sensed contact (Alex);
    # Runs without force sensing keep the original table.
    has_force = any(m.get("mean_contact_force_n") is not None for m in per_episode_metrics)
    lines += ["## Episodes", ""]
    header = "| seed | randomized | steps | final angle (deg) | success | termination |"
    rule = "|------|-----------|-------|-------------------|---------|-------------|"
    if has_force:
        header += " contact ticks | mean force (N) | max force (N) |"
        rule += "---------------|----------------|---------------|"
    lines += [header, rule]
    for m in per_episode_metrics:
        final_deg = (
            f"{math.degrees(m['final_door_angle_rad']):.1f}"
            if math.isfinite(m["final_door_angle_rad"])
            else "non-finite"
        )
        row = (
            f"| {m['seed']} | {'yes' if m['randomized'] else 'no'} | {m['n_steps']} "
            f"| {final_deg} | {'yes' if m['success'] else 'no'} "
            f"| {m['termination_reason']} |"
        )
        if has_force:
            row += (
                f" {m['contact_ticks']} | {_force_cell(m['mean_contact_force_n'])} "
                f"| {_force_cell(m['max_contact_force_n'])} |"
            )
        lines.append(row)

    lines += ["", "## Aggregate metrics", "", "```json", json.dumps(aggregate, indent=2), "```", ""]

    lines += ["## Dataset exports", ""]
    if exports:
        for action_space, directory in exports.items():
            lines.append(f"- `{action_space}` → `{directory}`")
    else:
        lines.append("- none")
    a1_note = _a1_status_note(episodes, exports)
    if a1_note:
        lines.append(a1_note)
    lines.append("")

    lines += ["## Plots", ""]
    for name, plot_path in (plots or {}).items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")

    lines += ["## Videos", ""]
    if videos and videos.get("files"):
        for video in videos["files"]:
            lines.append(f"- `{video}`")
    else:
        reason = (videos or {}).get("status", "not requested")
        lines.append(f"- none ({reason})")
    lines.append("")

    lines += ["## Limitations / placeholders", ""]
    lines += [f"- {item}" for item in limitations]
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def _a1_status_note(
    episodes: list[EpisodeBuffer], exports: dict[str, Path] | None
) -> str | None:
    """A1 status line: nothing when exported, otherwise say why it is missing."""
    from alexdoor_xas.action.spaces import A1_JOINT_DELTA

    if exports and A1_JOINT_DELTA in exports:
        return None
    has_joint_targets = bool(
        episodes and episodes[0].steps and "joint_pos_target" in episodes[0].steps[0].proprio
    )
    if has_joint_targets:
        return (
            f"- `{A1_JOINT_DELTA}`: **not exported in this run** — episodes record per-tick "
            "joint positions/velocities/targets, so A1 is relabelable "
            "(see knowledge/wiki/topics/action-representations-and-adapters.md)."
        )
    return f"- `{A1_JOINT_DELTA}`: **not exported** — joint targets were not recorded."


def _force_cell(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


__all__ = ["write_run_report"]
