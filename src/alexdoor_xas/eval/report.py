"""Scripted-run Markdown report."""

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
    exports: dict[str, Path],
    plots: dict[str, Path],
    videos: dict[str, Any],
    limitations: list[str],
) -> Path:
    """Write one scripted-run report."""
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

    lines += ["## Episodes", ""]
    lines += [
        "| seed | randomized | steps | final angle (deg) | success | termination | "
        "contact ticks | mean force (N) | max force (N) |",
        "|------|-----------|-------|-------------------|---------|-------------|"
        "---------------|----------------|---------------|",
    ]
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
        row += (
            f" {m['contact_ticks']} | {_force_cell(m['mean_contact_force_n'])} "
            f"| {_force_cell(m['max_contact_force_n'])} |"
        )
        lines.append(row)

    lines += ["", "## Aggregate metrics", "", "```json", json.dumps(aggregate, indent=2), "```", ""]

    lines += ["## Dataset exports", ""]
    for action_space, directory in exports.items():
        lines.append(f"- `{action_space}` → `{directory}`")
    if not exports:
        lines.append("- none")
    a1_note = _a1_status_note(exports)
    if a1_note:
        lines.append(a1_note)
    lines.append("")

    lines += ["## Plots", ""]
    for name, plot_path in plots.items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")

    lines += ["## Videos", ""]
    if videos.get("files"):
        for video in videos["files"]:
            lines.append(f"- `{video}`")
    else:
        reason = videos.get("status", "not requested")
        lines.append(f"- none ({reason})")
    lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in limitations]
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def _a1_status_note(exports: dict[str, Path]) -> str | None:
    from alexdoor_xas.action.spaces import A1_JOINT_DELTA

    if A1_JOINT_DELTA in exports:
        return None
    return (
        f"- `{A1_JOINT_DELTA}`: **not exported in this run** — recorded joint targets "
        "keep A1 relabelable."
    )


def _force_cell(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


__all__ = ["write_run_report"]
