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

DATA_ROOT_DEFAULT="${PHASE_DATA_ROOT:-/app/data}"
DATA_ROOT="$(prompt "PHASE data root (blank = env/default)" "${DATA_ROOT_DEFAULT}")"
PROJECT_ID="$(prompt "Project ID" "")"
SYSTEM_ID="$(prompt "System ID" "")"
META_IDS="$(prompt "Metastable IDs (comma separated)" "")"
MAX_CLUSTERS="$(prompt "Max clusters per residue" "6")"
MAX_FRAMES="$(prompt "Max cluster frames (0 = all)" "0")"
CONTACT_CUTOFF="$(prompt "Contact cutoff (A)" "10.0")"
CONTACT_MODE="$(prompt "Contact atom mode (CA/CM)" "CA")"
DENSITY_MAXK="$(prompt "Density maxk" "100")"
DENSITY_Z="$(prompt "Density Z (auto/number)" "auto")"
ASSIGN_LABELS="$(prompt "Compute assigned labels? (Y/n)" "Y")"

if [ -z "$PROJECT_ID" ] || [ -z "$SYSTEM_ID" ] || [ -z "$META_IDS" ]; then
  echo "Project ID, System ID, and Metastable IDs are required."
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.cluster_npz
  --project-id "$PROJECT_ID"
  --system-id "$SYSTEM_ID"
  --metastable-ids "$META_IDS"
  --max-clusters-per-residue "$MAX_CLUSTERS"
  --contact-cutoff "$CONTACT_CUTOFF"
  --contact-atom-mode "$CONTACT_MODE"
  --density-maxk "$DENSITY_MAXK"
  --density-z "$DENSITY_Z"
)

if [ -n "$DATA_ROOT" ]; then
  CMD+=(--data-root "$DATA_ROOT")
fi

if [ "$MAX_FRAMES" != "0" ] && [ -n "$MAX_FRAMES" ]; then
  CMD+=(--max-cluster-frames "$MAX_FRAMES")
fi

ASSIGN_LABELS="$(printf "%s" "$ASSIGN_LABELS" | tr '[:upper:]' '[:lower:]')"
if [ "$ASSIGN_LABELS" = "n" ] || [ "$ASSIGN_LABELS" = "no" ]; then
  CMD+=(--no-assign)
fi

echo "Running clustering..."
PHASE_DATA_ROOT="$DATA_ROOT" PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"
