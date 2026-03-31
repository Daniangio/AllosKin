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
    --root) OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id) OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id) OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id) CLUSTER_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  VENV_DIR="${VIRTUAL_ENV}"
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
if [ -z "$OFFLINE_PROJECT_ID" ]; then offline_select_project; fi
if [ -z "$OFFLINE_SYSTEM_ID" ]; then offline_select_system; fi
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
SELECT_NONE="none|Skip energy evaluation"
MODEL_ROW="$(offline_choose_one "Select Potts model for energies (blank = comparisons only):" "$SELECT_NONE
$MODEL_LINES")"
MODEL_ID="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $1}')"
if [ "$MODEL_ID" = "none" ]; then
  MODEL_ID=""
fi

MD_LABEL_MODE="$(prompt "MD label mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then MD_LABEL_MODE="assigned"; fi
KEEP_INVALID="$(prompt "Keep invalid SA rows? (y/N)" "N")"
KEEP_INVALID="$(printf "%s" "$KEEP_INVALID" | tr '[:upper:]' '[:lower:]')"
WORKERS="$(prompt "Workers (0=auto)" "0")"
SHOW_PROGRESS="$(prompt "Show progress? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_analyze
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --md-label-mode "$MD_LABEL_MODE"
  --workers "$WORKERS"
)
if [ -n "$MODEL_ID" ]; then CMD+=(--model "$MODEL_ID"); fi
if [ "$KEEP_INVALID" = "y" ] || [ "$KEEP_INVALID" = "yes" ]; then CMD+=(--keep-invalid); fi
if [ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ]; then CMD+=(--progress); fi

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
PHASE_DATA_ROOT="$OFFLINE_ROOT" "${CMD[@]}"
