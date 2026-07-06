#!/usr/bin/env python
"""No-network W&B scaffold check.

Default mode is disabled. Use ``--mode offline`` to verify that W&B can create
local run metadata under an ignored outputs directory. This script never handles
or prints W&B API keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alexdoor_xas.tracking import load_wandb_config, start_wandb_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AlexDoor-XAS W&B setup")
    parser.add_argument("--config", type=Path, default=None, help="W&B YAML config path")
    parser.add_argument("--mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--project", default=None, help="W&B project override")
    parser.add_argument("--entity", default=None, help="W&B entity override")
    parser.add_argument("--dir", type=Path, default=None, help="local W&B output directory")
    parser.add_argument("--name", default=None, help="run name override")
    parser.add_argument("--group", default=None, help="run group override")
    parser.add_argument("--job-type", default="setup-check", help="run job_type override")
    parser.add_argument("--tag", action="append", default=None, help="tag; repeatable")
    parser.add_argument(
        "--log-artifact",
        action="store_true",
        help="log a tiny generated JSON artifact; never logs real datasets/checkpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_wandb_config(
        args.config,
        overrides={
            "mode": args.mode,
            "project": args.project,
            "entity": args.entity,
            "dir": args.dir,
            "name": args.name,
            "group": args.group,
            "job_type": args.job_type,
            "tags": args.tag,
            "log_artifacts": args.log_artifact,
        },
    )
    payload = {
        "check": "wandb",
        "mode": cfg.mode,
        "project": cfg.project,
        "job_type": cfg.job_type,
        "log_artifacts": cfg.log_artifacts,
    }
    with start_wandb_run(cfg, config=payload) as run:
        run.log({"wandb_check/status": 1})
        if args.log_artifact:
            artifact_dir = cfg.dir / "check_payload"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "wandb_check.json"
            artifact_path.write_text(json.dumps(payload, indent=2) + "\n")
            run.log_file_artifact(
                artifact_path,
                name="wandb-check",
                type="diagnostic",
                metadata={"mode": cfg.mode},
                allow=True,
            )
    print(
        f"PASS: W&B check mode={cfg.mode} project={cfg.project} "
        f"dir={cfg.dir} artifacts={cfg.log_artifacts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
