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

DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"

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

trim() {
  printf "%s" "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
CLUSTER_ID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id)
      OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id)
      OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id)
      CLUSTER_ID="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

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

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "${DEFAULT_ROOT}"
else
  OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"
  export PHASE_DATA_ROOT="$OFFLINE_ROOT"
fi

if [ -z "$OFFLINE_PROJECT_ID" ]; then
  offline_select_project
fi

if [ -z "$OFFLINE_SYSTEM_ID" ]; then
  offline_select_system
fi

if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
if [ -z "$CLUSTER_ID" ]; then
  echo "No cluster selected."
  exit 1
fi

MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
if [ -z "$(trim "$MODEL_LINES")" ]; then
  echo "No Potts models found for cluster: $CLUSTER_ID"
  exit 1
fi

echo ""
echo "Lambda model endpoints:"
echo "  - Model B corresponds to λ=0"
echo "  - Model A corresponds to λ=1"
echo ""
MODEL_B_ROW="$(offline_choose_one "Select endpoint model B (λ=0):" "$MODEL_LINES")"
MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
MODEL_B_NAME="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $2}')"
MODEL_A_ROW="$(offline_choose_one "Select endpoint model A (λ=1):" "$MODEL_LINES")"
MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
MODEL_A_NAME="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $2}')"
if [ -z "$MODEL_A_ID" ] || [ -z "$MODEL_B_ID" ]; then
  echo "Both endpoint models are required."
  exit 1
fi
if [ "$MODEL_A_ID" = "$MODEL_B_ID" ]; then
  echo "Endpoint models must be different."
  exit 1
fi

LAM="$(prompt "Lambda (0..1)" "0.5")"
DEFAULT_NAME="Lambda ${LAM} ${MODEL_B_NAME:-$MODEL_B_ID} -> ${MODEL_A_NAME:-$MODEL_A_ID}"
MODEL_NAME="$(prompt "Model name" "$DEFAULT_NAME")"

GAUGE="$(prompt "Zero-sum gauge endpoints? (Y/n)" "Y")"
GAUGE="$(printf "%s" "$GAUGE" | tr '[:upper:]' '[:lower:]')"
NO_GAUGE=""
if [ "$GAUGE" = "n" ] || [ "$GAUGE" = "no" ]; then
  NO_GAUGE="--no-gauge"
fi

exec "$PYTHON_BIN" -m phase.scripts.potts_lambda_model \
  --project-id "$OFFLINE_PROJECT_ID" \
  --system-id "$OFFLINE_SYSTEM_ID" \
  --cluster-id "$CLUSTER_ID" \
  --model-a-id "$MODEL_A_ID" \
  --model-b-id "$MODEL_B_ID" \
  --lam "$LAM" \
  --name "$MODEL_NAME" \
  $NO_GAUGE

