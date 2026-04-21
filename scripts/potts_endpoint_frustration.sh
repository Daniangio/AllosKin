#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

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
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  echo "No active virtual environment detected." >&2
  echo "Activate .venv-phase or .venv-potts-fit first." >&2
  exit 1
fi

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "$DEFAULT_ROOT"
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
  echo "No cluster selected." >&2
  exit 1
fi

MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
if [ -z "$(trim "$MODEL_LINES")" ]; then
  echo "No Potts models found for cluster: $CLUSTER_ID" >&2
  exit 1
fi

MODEL_A_ROW="$(offline_choose_one "Select model A:" "$MODEL_LINES")"
MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
MODEL_B_ROW="$(offline_choose_one "Select model B:" "$MODEL_LINES")"
MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
if [ -z "$MODEL_A_ID" ] || [ -z "$MODEL_B_ID" ]; then
  echo "Model A and model B are required." >&2
  exit 1
fi
if [ "$MODEL_A_ID" = "$MODEL_B_ID" ]; then
  echo "Model A and model B must be different." >&2
  exit 1
fi

SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
if [ -z "$(trim "$SAMPLE_LINES")" ]; then
  echo "No samples found for this cluster." >&2
  exit 1
fi

SAMPLE_ROWS="$(offline_choose_multi "Select trajectories to analyze:" "$SAMPLE_LINES")"
SAMPLE_IDS="$(
  printf "%s\n" "$SAMPLE_ROWS" \
    | awk -F'|' '{print $4}' \
    | sed '/^[[:space:]]*$/d' \
    | while read -r p; do
        p="$(trim "$p")"
        [ -z "$p" ] && continue
        basename "$(dirname "$p")"
      done \
    | paste -sd',' -
)"
if [ -z "$SAMPLE_IDS" ]; then
  echo "No samples selected." >&2
  exit 1
fi

MD_LABEL_MODE="$(prompt "MD labels mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then MD_LABEL_MODE="assigned"; fi
KEEP_INVALID="false"
if prompt_bool "Keep invalid frames? (y/N)" "N"; then KEEP_INVALID="true"; fi
TOP_K_EDGES="$(prompt "Top edges to store for per-edge frustration" "2000")"
WORKERS="$(prompt "Workers (0=auto)" "0")"
SHOW_PROGRESS="$(prompt "Show progress? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_endpoint_frustration
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-a-id "$MODEL_A_ID"
  --model-b-id "$MODEL_B_ID"
  --sample-ids "$SAMPLE_IDS"
  --md-label-mode "$MD_LABEL_MODE"
  --top-k-edges "$TOP_K_EDGES"
  --workers "$WORKERS"
)

if [ "$KEEP_INVALID" = "true" ]; then CMD+=(--keep-invalid); fi
if [ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ]; then CMD+=(--progress); fi

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
