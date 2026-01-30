#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"
if [ -d "${ROOT_DIR}/.venv-potts-fit" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_RESULTS="${ROOT_DIR}/results/potts_delta_fit_${TIMESTAMP}"

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

prompt_bool() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  var="${var:-$default}"
  var="$(printf "%s" "$var" | tr '[:upper:]' '[:lower:]')"
  case "$var" in
    y|yes|true|1) return 0 ;;
    *) return 1 ;;
  esac
}

trim() {
  printf "%s" "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
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

offline_prompt_root "${ROOT_DIR}/data"
offline_select_project
offline_select_system
MODEL_ROW="$(offline_select_model)"
BASE_MODEL="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $3}')"
CLUSTER_ID="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $4}')"
if [ -z "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL" ]; then
  echo "Base model not found: $BASE_MODEL"
  exit 1
fi

RESULTS_DIR="$(prompt "Results directory" "${DEFAULT_RESULTS}")"
MODEL_NAME="$(prompt "Model name (base for delta models)" "")"

NPZ_PATH=""
STATE_IDS=""

CLUSTER_ROW="$(_offline_list list-clusters --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid {print; exit}')"
if [ -z "$CLUSTER_ROW" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
NPZ_PATH="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $3}')"
if [ -z "$NPZ_PATH" ] || [ ! -f "$NPZ_PATH" ]; then
  echo "Cluster NPZ file is required."
  exit 1
fi

STATE_ROWS="$(offline_select_analysis_states)"
STATE_IDS="$(printf "%s\n" "$STATE_ROWS" | awk -F'|' '{print $1}' | paste -sd, -)"
if [ -z "$STATE_IDS" ]; then
  echo "Select at least one state ID."
  exit 1
fi

EPOCHS="$(prompt "Epochs" "200")"
LR="$(prompt "Learning rate" "1e-3")"
LR_MIN="$(prompt "Learning rate min" "1e-3")"
LR_SCHEDULE="$(prompt "Learning rate schedule (cosine/none)" "cosine")"
BATCH_SIZE="$(prompt "Batch size" "512")"
SEED="$(prompt "Random seed" "0")"
DEVICE="$(prompt "Device (auto/cuda/cpu)" "auto")"
DEVICE="$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')"

DELTA_L2="$(prompt "Delta L2 weight" "0.0")"
DELTA_GROUP_H="$(prompt "Delta group sparsity (fields)" "0.0")"
DELTA_GROUP_J="$(prompt "Delta group sparsity (couplings)" "0.0")"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_delta_fit
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --base-model "$BASE_MODEL"
  --results-dir "$RESULTS_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --lr-min "$LR_MIN"
  --lr-schedule "$LR_SCHEDULE"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --delta-l2 "$DELTA_L2"
  --delta-group-h "$DELTA_GROUP_H"
  --delta-group-j "$DELTA_GROUP_J"
)

if [ -n "$MODEL_NAME" ]; then
  CMD+=(--model-name "$MODEL_NAME")
fi

if [ "$DEVICE" != "auto" ] && [ -n "$DEVICE" ]; then
  CMD+=(--device "$DEVICE")
fi

CMD+=(--npz "$NPZ_PATH" --state-ids "$STATE_IDS")

if prompt_bool "Skip saving combined models? (y/N)" "N"; then
  CMD+=(--no-combined)
fi

echo "Running delta Potts fit..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Outputs in: ${RESULTS_DIR}"
