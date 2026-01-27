#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "${ROOT_DIR}/.venv-potts-fit" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
fi

prompt() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  if [ -z "$var" ]; then
    echo "$default"
  else
    echo "$var"
  fi
}

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  VENV_DIR="${VIRTUAL_ENV}"
  echo "Using active virtual environment at: ${VENV_DIR}"
else
  echo "No active virtual environment detected."
  if [ -x "${DEFAULT_ENV}/bin/python" ]; then
    echo "Activate it with: source ${DEFAULT_ENV}/bin/activate"
  else
    echo "Create one first: scripts/potts_setup.sh"
  fi
  exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

DESCRIPTORS="$(prompt "Descriptor NPZ paths (comma separated)" "")"
if [ -z "$DESCRIPTORS" ]; then
  echo "At least one descriptor NPZ path is required."
  exit 1
fi
EVAL_DESCRIPTORS="$(prompt "Eval-only NPZ paths (comma separated, optional)" "")"
N_JOBS="$(prompt "Worker processes (0 = all cpus)" "1")"
DENSITY_Z="$(prompt "Density z (auto or float)" "auto")"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.cluster_npz
  --descriptors "$DESCRIPTORS"
  --n-jobs "$N_JOBS"
  --density-z "$DENSITY_Z"
)

if [ -n "$EVAL_DESCRIPTORS" ]; then
  CMD+=(--eval-descriptors "$EVAL_DESCRIPTORS")
fi

echo "Running clustering..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"
