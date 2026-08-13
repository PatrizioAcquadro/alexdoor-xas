"""Scripted run orchestration from episodes to reports and datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
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
    check_alex_episode,
    contact_force_summary,
)
from alexdoor_xas.policies.scripted import DoorPushControllerCfg, VariationBounds
from alexdoor_xas.recording import EpisodeBuffer, write_episode

VIDEO_FPS = 60


@dataclass
class RunArtifacts:
    run_dir: Path
    episodes: list[EpisodeBuffer]
    per_episode_metrics: list[dict[str, Any]]
    aggregate: dict[str, Any]
    exports: dict[str, Path]
    plots: dict[str, Path]
    videos: dict[str, Any]
    report_path: Path
    limitations: list[str]
    sanity: dict[str, Any]


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
    engine_cfg: DataEngineCfg,
    controller_cfg: DoorPushControllerCfg | None = None,
    variation_bounds: VariationBounds | None = None,
    video: bool = False,
    export: bool = True,
    dataset_version: str = "v0",
) -> RunArtifacts:
    """Generate, validate, export, evaluate, and report one baseline run."""
    # A posed run must not replace the maintained D0 dataset version.
    non_default_pose = engine_cfg.door_pose_id != "D0"
    if export and non_default_pose:
        raise RuntimeError(
            "refusing to export datasets from a run with a non-default door pose "
            f"(door_pose_id={engine_cfg.door_pose_id!r}); rerun with "
            "export disabled (--no-export / run.export=false)"
        )

    run_dir = Path(outputs_root) / experiment / run_id
    _fresh_run_dir(run_dir)
    active_plan = plan_episodes(n_fixed, n_randomized, base_seed, bounds=variation_bounds)
    _write_run_config(
        run_dir,
        engine_cfg,
        controller_cfg,
        n_fixed,
        n_randomized,
        base_seed,
        dataset_version,
    )
    env_tick_limit = getattr(env, "max_episode_length", None)
    if env_tick_limit is not None and engine_cfg.max_ticks > int(env_tick_limit):
        raise RuntimeError(
            f"engine max_ticks ({engine_cfg.max_ticks}) exceeds the env's episode "
            f"length ({int(env_tick_limit)} control ticks): steps past the env "
            "budget would silently record post-auto-reset state"
        )
    episodes_dir = run_dir / "episodes"
    videos_state: dict[str, Any] = {
        "status": "enabled" if video else "not requested",
        "files": [],
    }

    episodes: list[EpisodeBuffer] = []
    for item in active_plan:
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
    if sanity["n_episodes_with_warnings"]:
        limitations.append(
            f"Sanity warnings on {sanity['n_episodes_with_warnings']} episode(s) "
            f"(see metrics/sanity.json)"
        )
    if sanity["n_episodes_with_errors"]:
        failing = [entry["seed"] for entry in sanity["episodes"] if entry["errors"]]
        raise RuntimeError(
            f"sanity checks failed on {sanity['n_episodes_with_errors']} episode(s) "
            f"(seeds {failing}); run aborted before export — see "
            f"{metrics_dir / 'sanity.json'}"
        )
    exports = export_datasets(episodes, datasets_root, version=dataset_version) if export else {}
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


def _run_sanity_checks(episodes: list[EpisodeBuffer], metrics_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for episode in episodes:
        result = check_alex_episode(episode)
        entries.append(
            {
                "episode_id": episode.meta.episode_id,
                "seed": episode.meta.seed,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "force": contact_force_summary(episode),
            }
        )
    summary = {
        "n_episodes_checked": len(entries),
        "n_episodes_with_errors": sum(1 for entry in entries if entry["errors"]),
        "n_episodes_with_warnings": sum(1 for entry in entries if entry["warnings"]),
        "episodes": entries,
    }
    (metrics_dir / "sanity.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _fresh_run_dir(run_dir: Path) -> None:
    """Remove only run-owned artifacts from an earlier run with the same ID."""

    for subdir in ("episodes", "videos", "metrics", "plots", "logs"):
        path = run_dir / subdir
        if path.exists():
            shutil.rmtree(path)
    report = run_dir / "report.md"
    if report.exists():
        report.unlink()


def _make_render_hook(env, frames: list[Any], videos_state: dict[str, Any]):
    def hook(_tick: int) -> None:
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
    dataset_version: str,
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_config.json").write_text(
        json.dumps(
            {
                "engine_cfg": engine_cfg.to_dict(),
                "controller_cfg": asdict(controller_cfg or DoorPushControllerCfg()),
                "n_fixed": n_fixed,
                "n_randomized": n_randomized,
                "base_seed": base_seed,
                "dataset_version": dataset_version,
            },
            indent=2,
        )
        + "\n"
    )
