#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

usage() {
  echo "Usage: $0 --depot-root PATH --scratch-root PATH --repo-root PATH \\" >&2
  echo "  --manifest PATH --torch-spec SPEC --torch-index-url URL [--conda EXE]" >&2
}

DEPOT_ROOT=""
SCRATCH_ROOT=""
REPO_ROOT=""
MANIFEST=""
TORCH_SPEC=""
TORCH_INDEX_URL=""
CONDA_EXE="${CONDA_EXE:-conda}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --depot-root) DEPOT_ROOT="${2:-}"; shift 2 ;;
    --scratch-root) SCRATCH_ROOT="${2:-}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:-}"; shift 2 ;;
    --manifest) MANIFEST="${2:-}"; shift 2 ;;
    --torch-spec) TORCH_SPEC="${2:-}"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="${2:-}"; shift 2 ;;
    --conda) CONDA_EXE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in DEPOT_ROOT SCRATCH_ROOT REPO_ROOT MANIFEST TORCH_SPEC TORCH_INDEX_URL; do
  [[ -n "${!name}" ]] || { echo "Missing required value: $name" >&2; usage; exit 2; }
done
[[ -d "$DEPOT_ROOT" ]] || { echo "Depot root does not exist: $DEPOT_ROOT" >&2; exit 10; }
[[ -w "$DEPOT_ROOT" ]] || { echo "Depot root is not writable: $DEPOT_ROOT" >&2; exit 11; }
[[ -d "$SCRATCH_ROOT" ]] || { echo "Scratch root does not exist: $SCRATCH_ROOT" >&2; exit 12; }
[[ -w "$SCRATCH_ROOT" ]] || { echo "Scratch root is not writable: $SCRATCH_ROOT" >&2; exit 13; }
[[ -d "$REPO_ROOT/.git" ]] || { echo "Repository checkout is missing: $REPO_ROOT" >&2; exit 14; }
[[ -f "$REPO_ROOT/environment/gilbreth_pilot_py311.yml" ]] || { echo "Environment spec is missing" >&2; exit 15; }
[[ -f "$MANIFEST" ]] || { echo "Pilot manifest is missing: $MANIFEST" >&2; exit 16; }
command -v "$CONDA_EXE" >/dev/null || { echo "Conda executable is unavailable: $CONDA_EXE" >&2; exit 17; }

if [[ "$TORCH_INDEX_URL" =~ ://[^/@]+:[^/@]+@ ]]; then
  echo "Credential-bearing PyTorch index URLs are forbidden" >&2
  exit 18
fi
if [[ "$TORCH_SPEC" != torch==* ]]; then
  echo "--torch-spec must be an explicit torch==VERSION/BUILD requirement" >&2
  exit 19
fi

ENV_PREFIX="$DEPOT_ROOT/envs/alexdoor-gilbreth-pilot-py311"
INVENTORY_DIR="$DEPOT_ROOT/alexdoor-xas/cluster_pilot_n50/environment_bootstrap"
mkdir -p "$(dirname "$ENV_PREFIX")" "$INVENTORY_DIR"

if [[ -d "$ENV_PREFIX" ]]; then
  "$CONDA_EXE" env update --prefix "$ENV_PREFIX" \
    --file "$REPO_ROOT/environment/gilbreth_pilot_py311.yml" --prune
else
  "$CONDA_EXE" env create --prefix "$ENV_PREFIX" \
    --file "$REPO_ROOT/environment/gilbreth_pilot_py311.yml"
fi

"$CONDA_EXE" run --prefix "$ENV_PREFIX" python -m pip install \
  --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
"$CONDA_EXE" run --prefix "$ENV_PREFIX" python -m pip install \
  --no-deps --editable "$REPO_ROOT"
"$CONDA_EXE" run --prefix "$ENV_PREFIX" python -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'

cd "$REPO_ROOT"
"$CONDA_EXE" run --prefix "$ENV_PREFIX" python scripts/preflight_cluster_pilot.py \
  --config configs/cluster_pilot_n50.v1.json \
  --manifest "$MANIFEST" \
  --scratch-output "$SCRATCH_ROOT" \
  --report "$INVENTORY_DIR/preflight_report.json" \
  --environment-dir "$INVENTORY_DIR"

echo "PASS: persistent Gilbreth pilot environment bootstrapped at $ENV_PREFIX"
echo "Resolved inventory and lock: $INVENTORY_DIR"
