# Gilbreth N50 Compatibility Pilot

Gilbreth is a **non-Isaac training environment**. Do not install, import, or run Isaac Sim or
Isaac Lab there. Ubuntu remains the authority for dataset generation and all closed-loop
evaluation.

This workflow transfers the existing `door_push_alex_v2/v2_pose` compatibility dataset only:

- A2: `A2_ee_delta` (ACT pilot cell);
- A3: `A3_obj_rel_ee_delta` (Diffusion pilot cell);
- 50 total episodes with the shared 38 train / 6 validation / 6 test split;
- `dataset.obs_preset=core_door_pose`, seed 0, two epochs, validation every epoch, at most two
  training episodes, CUDA required, and W&B explicitly offline.

The historical stabilization manifest at
`outputs/local_smoke_n50/cluster_transfer_manifest.json` is preserved evidence. It is not the
pilot-transfer manifest and must not be modified or used for this pilot.

## Verified pilot evidence

The compatibility workflow completed end to end on Gilbreth on 2026-07-13 from commit
`10ba63ea1bc93501e9073683d9a2ce4f3416f393` (Slurm array `11279452`):

- ACT-A2 and Diffusion-A3 both completed on A100 80GB GPUs;
- the verified environment used Python 3.11.15, NumPy 2.4.6, PyTorch 2.12.1+cu126, CUDA 12.6,
  and Ruff 0.15.3;
- the symlink-free return package contained 52 payload files and passed all hash checks;
- the original durable W&B tree contained scratch-backed log symlinks, so this historical attempt
  required one-time symlink materialization and `latest-run` omission during return staging;
- Ubuntu loaded both returned best checkpoints successfully on CPU.

The ignored Ubuntu evidence is under `outputs/cluster_pilot_n50/returned/11279452/`. This proves
the non-Isaac training and return path for the compatibility pilot only; it does not authorize or
validate the later dataset-scale sweep.

The automatic symlink-free W&B publisher was merged after this historical attempt. It is locally
validated but still requires one live two-cell Gilbreth canary with direct return packaging before
the full sweep is prepared for execution.

## Required user-supplied Gilbreth values

Fill these from the live Gilbreth account and allocation documentation:

```bash
export GILBRETH_USER='<user>'
export GILBRETH_HOST='<host>'
export SLURM_ACCOUNT='<account>'
export SLURM_PARTITION='<partition>'
export DEPOT_ROOT='<absolute-depot-root>'
export SCRATCH_ROOT='<absolute-scratch-root>'
export TORCH_SPEC='<torch==version-or-build-supported-by-live-Gilbreth-driver>'
export TORCH_INDEX_URL='<official-PyTorch-index-for-that-supported-build>'
```

Set a QOS only if the selected partition/account documentation explicitly requires or supports
one. Add `--qos '<qos>'` to the renderer command in that case. Add `--require-a100-80gb` only when
the requested allocation is specifically an A100-80GB allocation.

Never put credentials, W&B keys, authenticated URLs, `.netrc`, or private keys in the repository,
pilot config, manifest, or rendered Slurm script.

## 1. Ubuntu: build the committed pilot-transfer package

The builder requires a clean tree and binds every file to the exact current commit. Run only after
the local validation commit exists:

```bash
cd /home/pacquadr/Desktop/DoorManipulation
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_cluster_pilot_manifest.py build \
  --config configs/cluster_pilot_n50.v1.json \
  --output-dir outputs/cluster_pilot_n50
```

Generated ignored artifacts:

- `outputs/cluster_pilot_n50/pilot_transfer_manifest.json`;
- `outputs/cluster_pilot_n50/rsync-files.txt`;
- `outputs/cluster_pilot_n50/rsync-command.txt`.

The resumable outbound command uses checksum comparison instead of append semantics:

```bash
rsync -avP --partial --checksum \
  --files-from=outputs/cluster_pilot_n50/rsync-files.txt \
  ./ <user>@<host>:<remote_root>/
```

Use the Gilbreth source checkout path as `<remote_root>`. The verified setup uses the single
physical checkout at `/home/pacquadr/Desktop/DoorManipulation/`; the depot source path is a symlink
to that checkout, not a second clone. The exact source commit must be available to Gilbreth through
a separately user-approved Git publication step.

## 2. Gilbreth: exact source checkout and incoming verification

Define the derived paths. Keep one physical checkout and require the depot path used by rendered
jobs to resolve to it:

```bash
export REPO_ROOT=/home/pacquadr/Desktop/DoorManipulation
export DEPOT_SOURCE_LINK="$DEPOT_ROOT/alexdoor-xas/source"
export CONDA_PREFIX="$DEPOT_ROOT/envs/alexdoor-gilbreth-pilot-py311"
export DURABLE_RESULTS_ROOT="$DEPOT_ROOT/alexdoor-xas/cluster_pilot_n50/results"
export PILOT_SCRATCH_ROOT="$SCRATCH_ROOT/alexdoor-xas/cluster_pilot_n50"
export PILOT_MANIFEST="$REPO_ROOT/outputs/cluster_pilot_n50/pilot_transfer_manifest.json"
test "$(readlink -f "$DEPOT_SOURCE_LINK")" = "$REPO_ROOT"
```

Alex V2 provenance verification also needs the pure URDF at
`/home/pacquadr/Desktop/Alex/urdf/alex_v2.urdf`. It is not an Isaac dependency. Its required
SHA-256 is `7742b88d9cb81e80f3d1e5c1906e31f38ca03734085454505e550b24009920b3`.

Check out the exact `source_git.commit` printed by the local builder, then verify the transfer:

```bash
cd "$REPO_ROOT"
git checkout --detach '<source_git.commit>'
PYTHONPATH=$REPO_ROOT python scripts/build_cluster_pilot_manifest.py verify \
  --config configs/cluster_pilot_n50.v1.json \
  --manifest "$PILOT_MANIFEST"
```

Do not continue if the commit, clean-tree, inventory, count, fingerprint, size, hash, or secret
check fails.

## 3. Gilbreth: bootstrap the persistent Python 3.11 environment

First inspect live driver/module evidence (`nvidia-smi` and the applicable Gilbreth module/partition
documentation). Select `TORCH_SPEC` and `TORCH_INDEX_URL` from that evidence; there is deliberately
no guessed PyTorch/CUDA build in the repository.

```bash
cd "$REPO_ROOT"
bash scripts/bootstrap_gilbreth_pilot.sh \
  --depot-root "$DEPOT_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --repo-root "$REPO_ROOT" \
  --manifest "$PILOT_MANIFEST" \
  --torch-spec "$TORCH_SPEC" \
  --torch-index-url "$TORCH_INDEX_URL"
```

The bootstrap fails on missing paths, installation errors, non-Python-3.11 environments, present
Isaac namespaces, transfer verification failures, or checkpoint I/O failures. It writes a
credentials-free resolved environment inventory and package lock under depot.

## 4. Gilbreth: render and inspect the two-cell Slurm array

Create the Slurm log directory before submission, because Slurm opens those files before the job
script runs:

```bash
mkdir -p "$PILOT_SCRATCH_ROOT/slurm" "$DURABLE_RESULTS_ROOT"
cd "$REPO_ROOT"
"$CONDA_PREFIX/bin/python" scripts/render_cluster_pilot_slurm.py \
  --config configs/cluster_pilot_n50.v1.json \
  --manifest "$PILOT_MANIFEST" \
  --depot-root "$DEPOT_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --durable-results-root "$DURABLE_RESULTS_ROOT" \
  --account "$SLURM_ACCOUNT" \
  --partition "$SLURM_PARTITION" \
  --output outputs/cluster_pilot_n50/pilot.slurm
bash -n outputs/cluster_pilot_n50/pilot.slurm
```

The rendered `0-1%2` array uses Gilbreth's `#SBATCH --gpus-per-node=1` request. Account,
partition, optional QOS, memory, CPU, wall-time, and array concurrency remain configurable. Each
cell verifies the manifest, runs the pure plus live-CUDA preflight, invokes `scripts/train_act.py`
or `scripts/train_diffusion.py` directly, writes atomic completion/failure status, and atomically
publishes checkpoints, training logs, resolved configs, open-loop metrics, W&B offline data,
environment evidence, and cell logs from scratch to durable depot storage.

Scratch execution and durable publication are attempt-specific:

```text
attempts/<SLURM_ARRAY_JOB_ID>/<SLURM_ARRAY_TASK_ID>/<run_id>/
```

The nested array job ID, task ID, and run ID are recorded in every cell status. A retry submitted
as a new array receives a new attempt directory. A requeue or duplicate execution with an already
used identity fails instead of overwriting or reusing scratch or durable results. Do not delete or
rename an old attempt to make a retry pass; submit a new array and select its new array job ID for
the return workflow.

Submission command — **not executed by the local preparation task**:

```bash
sbatch outputs/cluster_pilot_n50/pilot.slurm
```

## 5. Gilbreth: build the durable return package

After both array cells have durable completion status, select exactly one array submission by the
numeric job ID printed by `sbatch` (the shared `SLURM_ARRAY_JOB_ID`, not a task job ID):

```bash
export PILOT_ATTEMPT_ID='<SLURM_ARRAY_JOB_ID>'
cd "$REPO_ROOT"
"$CONDA_PREFIX/bin/python" scripts/build_cluster_pilot_return_manifest.py build \
  --results-root "$DURABLE_RESULTS_ROOT" \
  --attempt-id "$PILOT_ATTEMPT_ID" \
  --config configs/cluster_pilot_n50.v1.json \
  --manifest "$PILOT_MANIFEST"
```

The builder requires the selected attempt to contain exactly task `0` with the ACT run and task
`1` with the Diffusion run. It rejects a missing selection, legacy run-ID-only layouts, mixed job
or task identities, stale status schemas, and any source-commit mismatch. Other attempt
directories are never selected implicitly.

The return builder also rejects every symlink. W&B may create `debug.log`, `debug-internal.log`,
`debug-core.log`, and `latest-run` links; some `debug-core.log` links may point back into scratch.
Do not weaken the return verifier. New jobs sanitize W&B inside the temporary durable publication
directory: validated in-tree file links become regular files, `latest-run` links are omitted, and
the final atomic move occurs only after the destination is confirmed symlink-free. Each published
W&B directory includes a credentials-free `publication_report.json` with the materialized paths
and hashes. Attempt `11279452` remains historical evidence of the earlier manual staging path; it
must not be modified.

The resumable checksum-based return template uses the remote file list directly:

```bash
rsync -avP --partial --checksum \
  --files-from=:<remote_results_root>/.pilot_return/attempts/<SLURM_ARRAY_JOB_ID>/return-files.txt \
  <user>@<host>:<remote_results_root>/ <local_return_root>/
```

## 6. Ubuntu: verify returned hashes and load both checkpoints

Run from the exact pilot source commit. This is a pure CPU checkpoint load and does not start
Isaac or perform closed-loop evaluation:

```bash
cd /home/pacquadr/Desktop/DoorManipulation
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/verify_returned_cluster_pilot.py \
  --results-root '<local_return_root>' \
  --attempt-id '<SLURM_ARRAY_JOB_ID>' \
  --manifest '<local_return_root>/.pilot_return/attempts/<SLURM_ARRAY_JOB_ID>/return_manifest.json' \
  --config configs/cluster_pilot_n50.v1.json
```

Only after returned hashes, the source commit, the resolved environment evidence, and both CPU
checkpoint loads pass should the later scientific dataset/sweep work be considered.

## Full nested dataset-scale sweep preparation

The full-sweep tooling is separate from the historical two-cell pilot. It uses
`configs/cluster_sweep.v1.json`, one physical `v3_scale_master`, four shared
logical views, eight train-only normalization files, and a stable 16-cell
array. N50/N100/N250/N500 mean the number of **training** episodes; every view
also uses the same 25 validation and 25 test episodes. The current `v2_pose`
dataset remains stabilization and pilot evidence and is not rewritten.

With real scale publication complete and all local gates passing, a clean
committed Ubuntu checkout may build and immediately re-verify the exact
transfer package:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_cluster_sweep_manifest.py build \
  --config configs/cluster_sweep.v1.json \
  --output-dir outputs/cluster_sweep
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_cluster_sweep_manifest.py verify \
  --config configs/cluster_sweep.v1.json \
  --manifest outputs/cluster_sweep/sweep_transfer_manifest.json
```

The generated `rsync-files.txt` is an exact checksummed inventory of the two
paired datasets, master/view/norm metadata, and cluster-required source. It
binds the clean source commit and live Alex V2 URDF hash but deliberately does
not transfer the machine-local URDF. Do not continue on a count, identity,
fingerprint, hash, secret, clean-tree, or asset mismatch.

On Gilbreth, verify the received manifest first and use the proven persistent
Python 3.11 prefix `envs/alexdoor-gilbreth-pilot-py311`; the config, transfer
metadata, preflight, and renderer must all agree on that exact relative prefix.
Render the array with explicit live account/partition/depot and scratch values.
The renderer defaults to `0-15%2`, requests one GPU per cell, invokes
`$CONDA_PREFIX/bin/python` directly even under a polluted host `PATH`, forbids
Isaac and distributed launchers, and keeps all W&B runs offline. Each cell
verifies the transfer, runs pure and live-CUDA preflight, trains one exact
policy/space/view mapping, and atomically publishes symlink-free evidence from
scratch to:

```text
attempts/<SLURM_ARRAY_JOB_ID>/<SLURM_ARRAY_TASK_ID>/<run_id>/
```

After all 16 cells complete, select one explicit numeric array job ID and build
its return package with `scripts/build_cluster_sweep_return_manifest.py`. The
builder rejects partial/failed cells, mixed or duplicate identities, symlinks,
unexpected files, stale source/view/norm provenance, and hash drift. Ubuntu
then runs `scripts/verify_returned_cluster_sweep.py`, which verifies the exact
return inventory and CPU-loads every `best.pt` checkpoint. This verification
does not start Isaac or closed-loop evaluation.

The remote `return-files.txt` is directly usable with `rsync --files-from` from
the durable results root. It contains every payload path plus exactly one
attempt-local `return_manifest.json` and exactly one attempt-local
`return-files.txt`; both the cluster-side builder and Ubuntu verifier reject a
missing, duplicated, malformed, or mixed-attempt control path.

Historical pilot attempts `11279452` and `11279800` remain immutable evidence.
At this documentation state the full sweep has not been transferred or
submitted, and Phase 4 has not started.
