# Provenance and Artifact Lifecycle

AlexDoor-XAS treats calibration, data, views, normalization, checkpoints,
evaluations, and cluster packages as linked content identities. A downstream
artifact is valid only when all expected upstream identities match.

## Identity Layers

- **Source** — exact Git commit and, for transfer packages, a SHA-256 file
  inventory.
- **Robot** — raw Alex V2 asset inventory plus the derived runtime contract.
- **Calibration** — self-fingerprinted robot/task/controller measurements.
- **Dataset** — schema, representation, episode content, physical-master
  identity, and robot/calibration bindings.
- **Split and view** — exact membership, content grouping, ordering, and
  selection hash.
- **Normalization** — values computed only from the declared training IDs.
- **Checkpoint** — weights plus model, dataset, split/view, normalization,
  source, and robot identities.
- **Evaluation** — checkpoint set, simulator/runtime, matched seed plan,
  protocol, per-rollout evidence, and summary.

Validation is fail-closed. An unexplained mismatch must be diagnosed; expected
hashes must not be refreshed merely to make a changed artifact pass.

## Dataset Lifecycle

Reusable data lives under `datasets/<task>/<action_space>/<version>/`.
Generation owns a version. Re-exporting that version replaces its owned
contents, so behavior-changing generation requires a new version and refreshed
splits and training-only normalization. The scale path stages and atomically
publishes the paired A2/A3 products from one physical master.

Generated HDF5, model weights, and large runtime products remain ignored.
Models use the canonical dataset API, not direct raw keys.

## Run and Curated Outputs

Every execution run belongs under `outputs/<experiment>/<run_id>/` with its own
metrics, plots, videos, checkpoints, captures, and logs. Raw run directories
remain ignored.

Only small review artifacts—Markdown, JSON, CSV, PNG, SVG, or text—are promoted
to `outputs/curated/`. A curated package is immutable historical evidence. A
new protocol or rerun receives a new identity; it does not overwrite the old
package.

## Cluster Transfer and Return

Cluster packages use exact inventories and attempt-specific paths. A transfer
contains the intended source and required data/configuration identities.
Preflight validates them before Slurm rendering. Each attempt owns its logs,
checkpoints, metrics, environment capture, and portable tracking outputs.

Returns are symlink-free and inventory-verified. Workstation verification
requires every expected cell, no unexpected payload, correct attempt/source
identity, and checkpoint loading. Files from different attempts must never be
mixed. See [[decisions/fail-closed-provenance-and-immutable-artifacts|Fail-Closed Provenance and Immutable Artifacts]].

## Authority Boundaries

Git is the detailed historical record. Phase artifacts explain intent and
closeout but do not override current code or tracked evidence. For current
project status, `docs/status.md` and verified curated evidence are stronger
than stale summary prose in README or portions of `docs/cluster.md`.

W&B is optional supplementary tracking. Repository-owned configurations,
inventories, checkpoints, and evidence packages remain the reproducibility
authority.

## Primary References

- `src/alexdoor_xas/dataset/validate.py`
- `src/alexdoor_xas/dataset/views.py`
- `src/alexdoor_xas/policies/act/checkpoint.py`
- `src/alexdoor_xas/policies/diffusion/checkpoint.py`
- `src/alexdoor_xas/cluster_pilot/transfer.py`
- `src/alexdoor_xas/cluster_sweep/returns.py`
- `src/alexdoor_xas/eval/phase3_unified.py`
- `outputs/README.md`
- `outputs/curated/phase3_unified_evaluation/report.md`

## Version Notes

- 2026-07-08 — Robot, calibration, dataset, and checkpoint identities were
  rebound to the calibrated Alex V2 contract.
- 2026-07-16 to 2026-07-18 — Exact full-sweep return and unified evaluation
  packages completed the end-to-end provenance chain.
