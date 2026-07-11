"""One-run orchestration: episodes -> outputs/<experiment>/<run_id>/ + datasets.

Used by both ``scripts/run_scripted_baseline.py`` (the engine CLI) and
``scripts/verify_scripted_baseline.py`` (the Phase 2 gate), so the gate
exercises exactly the code path that produces real datasets. No Isaac imports:
the env comes in already constructed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alexdoor_xas.data_engine.export import export_datasets
from alexdoor_xas.data_engine.generate import (
    DataEngineCfg,
    EpisodePlanItem,
    plan_episodes,
    run_episode,
)
from alexdoor_xas.eval.metrics import aggregate_metrics, episode_metrics
from alexdoor_xas.eval.plots import door_angle_plot, final_angle_plot
from alexdoor_xas.eval.report import write_run_report
from alexdoor_xas.eval.sanity import (
    FORCE_DATASET_LIMIT_N,
    check_alex_episode,
    contact_force_diagnostics,
)
from alexdoor_xas.policies.scripted import DoorPushControllerCfg, VariationBounds
from alexdoor_xas.recording import EpisodeBuffer, write_episode

VIDEO_FPS = 60  # one rendered frame per control tick at control_dt = 1/60 s


@dataclass
class RunArtifacts:
    """Everything one baseline run produced (paths + in-memory results)."""

    run_dir: Path
    episodes: list[EpisodeBuffer]
    per_episode_metrics: list[dict[str, Any]]
    aggregate: dict[str, Any]
    exports: dict[str, Path]
    plots: dict[str, Path]
    videos: dict[str, Any]
    report_path: Path
    limitations: list[str] = field(default_factory=list)
    sanity: dict[str, Any] | None = None


def run_baseline(
    env,
    *,
    outputs_root: str | Path,
    datasets_root: str | Path,
    experiment: str,
    run_id: str,
    n_fixed: int,
    n_randomized: int,
    base_seed: int,
    engine_cfg: DataEngineCfg | None = None,
    controller_cfg: DoorPushControllerCfg | None = None,
    variation_bounds: VariationBounds | None = None,
    video: bool = False,
    export: bool = True,
) -> RunArtifacts:
    """Generate, record, export, evaluate, and report one baseline run.

    Rerunning the same ``run_id`` replaces its artifacts: the run-owned
    subdirectories (episodes/videos/metrics/plots/logs) and report.md are
    removed up front so a rerun can never leave stale episode files behind.

    Force-sensing (Alex) episodes are sanity-checked (``eval/sanity.py``)
    before any dataset export: the per-episode summary is always written to
    ``metrics/sanity.json``, warnings are reported verbatim, and any sanity
    *error* aborts the run loudly — bad data can no longer reach ``datasets/``
    silently and fail only at the Phase 3.0 gate.

    ``export=False`` records the run under ``outputs/`` only and never writes
    ``datasets/`` — the mode multi-pose generation uses so partial per-pose
    passes cannot masquerade as an official dataset version.
    """
    engine_cfg = engine_cfg or DataEngineCfg()
    run_dir = Path(outputs_root) / experiment / run_id
    _fresh_run_dir(run_dir)
    episodes_dir = run_dir / "episodes"
    videos_state: dict[str, Any] = {
        "status": "enabled" if video else "not requested",
        "files": [],
    }

    episodes: list[EpisodeBuffer] = []
    for item in plan_episodes(n_fixed, n_randomized, base_seed, bounds=variation_bounds):
        frames: list[Any] = []
        render_hook = _make_render_hook(env, frames, videos_state) if video else None
        episode = run_episode(
            env, item, engine_cfg, controller_cfg=controller_cfg, render_hook=render_hook
        )
        episodes.append(episode)
        write_episode(episode, episodes_dir)
        if frames:
            _write_video(frames, run_dir / "videos", item, videos_state)

    per_episode = [episode_metrics(episode) for episode in episodes]
    aggregate = aggregate_metrics(per_episode)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"aggregate": aggregate, "episodes": per_episode}, indent=2) + "\n"
    )

    sanity = _run_sanity_checks(episodes, metrics_dir)
    limitations = list(engine_cfg.limitations)
    if sanity is not None and sanity["n_episodes_with_warnings"]:
        limitations.append(
            f"Sanity warnings on {sanity['n_episodes_with_warnings']} episode(s) "
            f"(see metrics/sanity.json)"
        )
    if sanity is not None and sanity["n_episodes_with_errors"]:
        failing = [
            entry["seed"] for entry in sanity["episodes"] if entry["errors"]
        ]
        raise RuntimeError(
            f"sanity checks failed on {sanity['n_episodes_with_errors']} episode(s) "
            f"(seeds {failing}); run aborted before export — see "
            f"{metrics_dir / 'sanity.json'}"
        )

    exports = export_datasets(episodes, datasets_root) if export else {}
    plots = {
        "door_angle_vs_time": door_angle_plot(episodes, run_dir / "plots" / "door_angle.png"),
        "final_door_angle": final_angle_plot(episodes, run_dir / "plots" / "final_angle.png"),
    }

    if video and videos_state["status"] != "enabled":
        limitations.append(f"Video capture degraded: {videos_state['status']}")

    report_path = write_run_report(
        run_dir / "report.md",
        episodes=episodes,
        per_episode_metrics=per_episode,
        aggregate=aggregate,
        exports=exports,
        plots=plots,
        videos=videos_state,
        limitations=limitations,
    )
    _write_run_config(run_dir, engine_cfg, controller_cfg, n_fixed, n_randomized, base_seed)

    return RunArtifacts(
        run_dir=run_dir,
        episodes=episodes,
        per_episode_metrics=per_episode,
        aggregate=aggregate,
        exports=exports,
        plots=plots,
        videos=videos_state,
        report_path=report_path,
        limitations=limitations,
        sanity=sanity,
    )


def _run_sanity_checks(
    episodes: list[EpisodeBuffer], metrics_dir: Path
) -> dict[str, Any] | None:
    """Sanity-check force-sensing (Alex) episodes; write metrics/sanity.json.

    Same episode condition the Phase 3.0 dataset gate uses (joint proprio
    present). Returns ``None`` for runs with no such episodes (proxy). The
    summary carries every warning/error message verbatim plus the anti-windup
    IK clamp telemetry per episode — warnings are reported, never suppressed.
    """
    entries: list[dict[str, Any]] = []
    for episode in episodes:
        if not episode.steps or "joint_pos" not in episode.steps[0].proprio:
            continue
        result = check_alex_episode(
            episode, force_error_n=FORCE_DATASET_LIMIT_N
        )
        entries.append(
            {
                "episode_id": episode.meta.episode_id,
                "seed": episode.meta.seed,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "force_diagnostics": contact_force_diagnostics(
                    episode, force_limit_n=FORCE_DATASET_LIMIT_N
                ),
                "ik_clamp_telemetry": episode.extras.get("ik_clamp_telemetry"),
            }
        )
    if not entries:
        return None
    summary = {
        "n_episodes_checked": len(entries),
        "n_episodes_with_errors": sum(1 for entry in entries if entry["errors"]),
        "n_episodes_with_warnings": sum(1 for entry in entries if entry["warnings"]),
        "episodes": entries,
    }
    (metrics_dir / "sanity.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _fresh_run_dir(run_dir: Path) -> None:
    """Remove the artifacts a previous run with the same run_id produced.

    Targeted (not ``rmtree(run_dir)``): sibling content such as a gate's
    ``*_datasets`` directory or user notes dropped next to the artifacts
    must survive a rerun.
    """
    import shutil

    for subdir in ("episodes", "videos", "metrics", "plots", "logs"):
        path = run_dir / subdir
        if path.exists():
            shutil.rmtree(path)
    report = run_dir / "report.md"
    if report.exists():
        report.unlink()


def _make_render_hook(env, frames: list[Any], videos_state: dict[str, Any]):
    def hook(tick: int) -> None:
        if videos_state["status"] != "enabled":
            return
        try:
            frame = env.render()
        except Exception as error:  # noqa: BLE001 - degrade to no-video, keep the run alive.
            videos_state["status"] = f"render failed: {error}"
            return
        if frame is None:
            videos_state["status"] = (
                "render returned no frames (launch with --enable_cameras and "
                "render_mode='rgb_array')"
            )
            return
        frames.append(frame)

    return hook


def _write_video(
    frames: list[Any], videos_dir: Path, item: EpisodePlanItem, videos_state: dict[str, Any]
) -> None:
    try:
        import imageio.v3 as iio
        import numpy as np

        videos_dir.mkdir(parents=True, exist_ok=True)
        path = videos_dir / f"episode_seed{item.seed}.mp4"
        iio.imwrite(path, np.stack(frames), fps=VIDEO_FPS)
        videos_state["files"].append(str(path))
    except Exception as error:  # noqa: BLE001 - degrade to no-video, keep the run alive.
        videos_state["status"] = f"video encode failed: {error}"


def _write_run_config(
    run_dir: Path,
    engine_cfg: DataEngineCfg,
    controller_cfg: DoorPushControllerCfg | None,
    n_fixed: int,
    n_randomized: int,
    base_seed: int,
) -> None:
    import dataclasses

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_config.json").write_text(
        json.dumps(
            {
                "engine_cfg": engine_cfg.to_dict(),
                "controller_cfg": dataclasses.asdict(
                    controller_cfg or DoorPushControllerCfg()
                ),
                "n_fixed": n_fixed,
                "n_randomized": n_randomized,
                "base_seed": base_seed,
            },
            indent=2,
        )
        + "\n"
    )


__all__ = ["VIDEO_FPS", "RunArtifacts", "run_baseline"]
