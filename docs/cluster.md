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

Use the Gilbreth source checkout path as `<remote_root>`. The exact source commit must be made
available to Gilbreth through a separately user-approved Git publication step; this preparation
workflow creates a local commit but does not push it.

## 2. Gilbreth: exact source checkout and incoming verification

Define the derived paths:

```bash
export REPO_ROOT="$DEPOT_ROOT/alexdoor-xas/source"
export CONDA_PREFIX="$DEPOT_ROOT/envs/alexdoor-gilbreth-pilot-py311"
export DURABLE_RESULTS_ROOT="$DEPOT_ROOT/alexdoor-xas/cluster_pilot_n50/results"
export PILOT_SCRATCH_ROOT="$SCRATCH_ROOT/alexdoor-xas/cluster_pilot_n50"
export PILOT_MANIFEST="$REPO_ROOT/outputs/cluster_pilot_n50/pilot_transfer_manifest.json"
```

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

## Future sweep contract (recorded only)

The later sweep uses one deterministic master pool, equal pose balance, paired A2/A3 exports from
the same source episodes, fixed validation/test episodes, and nested N50/N100/N250/N500 training
subsets. N means the number of **training** episodes. The current `v2_pose` dataset remains only
stabilization and compatibility-pilot evidence. This task does not generate those datasets or
start the 16-cell sweep.
