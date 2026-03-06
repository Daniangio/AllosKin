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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
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
echo "Endpoint models:"
echo "  - Model A: reference endpoint A"
echo "  - Model B: reference endpoint B"
echo ""

MODEL_A_ROW="$(offline_choose_one "Select endpoint model A:" "$MODEL_LINES")"
MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
MODEL_A_NAME="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $2}')"
if [ -z "$MODEL_A_ID" ]; then
  echo "No model A selected."
  exit 1
fi

MODEL_B_ROW="$(offline_choose_one "Select endpoint model B:" "$MODEL_LINES")"
MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
MODEL_B_NAME="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $2}')"
if [ -z "$MODEL_B_ID" ]; then
  echo "No model B selected."
  exit 1
fi
if [ "$MODEL_A_ID" = "$MODEL_B_ID" ]; then
  echo "Model A and model B must be different."
  exit 1
fi

export _PHASE_LC_ROOT="$OFFLINE_ROOT"
export _PHASE_LC_PROJECT="$OFFLINE_PROJECT_ID"
export _PHASE_LC_SYSTEM="$OFFLINE_SYSTEM_ID"
export _PHASE_LC_CLUSTER="$CLUSTER_ID"
MD_LINES="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from phase.services.project_store import ProjectStore

root = Path(os.environ["_PHASE_LC_ROOT"]) / "projects"
store = ProjectStore(base_dir=root)
system = store.get_system(os.environ["_PHASE_LC_PROJECT"], os.environ["_PHASE_LC_SYSTEM"])
cluster_id = os.environ["_PHASE_LC_CLUSTER"]
entry = next((c for c in (system.metastable_clusters or []) if str(c.get("cluster_id")) == cluster_id), None)
if not isinstance(entry, dict):
    raise SystemExit(0)
for sample in (entry.get("samples") or []):
    if str(sample.get("type")) != "md_eval":
        continue
    sid = str(sample.get("sample_id") or "").strip()
    if not sid:
        continue
    name = str(sample.get("name") or sid)
    state_id = str(sample.get("state_id") or "")
    print(f"{sid}|{name} [{state_id}]")
PY
)"
unset _PHASE_LC_ROOT _PHASE_LC_PROJECT _PHASE_LC_SYSTEM _PHASE_LC_CLUSTER

if [ -z "$(trim "$MD_LINES")" ]; then
  echo "No md_eval samples found for this cluster."
  exit 1
fi

START_ROW="$(offline_choose_one "Select starting MD sample:" "$MD_LINES")"
START_SAMPLE_ID="$(printf "%s" "$START_ROW" | awk -F'|' '{print $1}')"
START_SAMPLE_NAME="$(printf "%s" "$START_ROW" | awk -F'|' '{print $2}')"
if [ -z "$START_SAMPLE_ID" ]; then
  echo "No starting sample selected."
  exit 1
fi

REF_A_ROW="$(offline_choose_one "Optional reference MD for A (blank to auto):" "$MD_LINES")"
REF_A_ID="$(printf "%s" "$REF_A_ROW" | awk -F'|' '{print $1}')"
REF_B_ROW="$(offline_choose_one "Optional reference MD for B (blank to auto):" "$MD_LINES")"
REF_B_ID="$(printf "%s" "$REF_B_ROW" | awk -F'|' '{print $1}')"

CONSTRAINT_MODE="$(prompt "Constraint source mode (manual/delta_js_auto)" "manual")"
CONSTRAINT_MODE="$(printf "%s" "$CONSTRAINT_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$CONSTRAINT_MODE" != "delta_js_auto" ]; then
  CONSTRAINT_MODE="manual"
fi

CONSTRAINED=""
DELTA_JS_EXPERIMENT_ID=""
DELTA_JS_FILTER_SETUP_ID=""
DELTA_JS_FILTER_EDGE_ALPHA="0.75"
CONSTRAINT_DELTA_JS_SAMPLE_ID=""
CONSTRAINT_AUTO_TOPK="12"
CONSTRAINT_AUTO_EDGE_ALPHA="0.3"
CONSTRAINT_AUTO_EXCLUDE_SUCCESS="y"
if [ "$CONSTRAINT_MODE" = "manual" ]; then
  CONSTRAINED="$(prompt "Constrained residues (comma-separated indices/keys)" "")"
  CONSTRAINED="$(trim "$CONSTRAINED")"
  if [ -z "$CONSTRAINED" ]; then
    echo "Constrained residues are required in manual mode."
    exit 1
  fi
else
  export _PHASE_LC_ROOT="$OFFLINE_ROOT"
  export _PHASE_LC_PROJECT="$OFFLINE_PROJECT_ID"
  export _PHASE_LC_SYSTEM="$OFFLINE_SYSTEM_ID"
  export _PHASE_LC_CLUSTER="$CLUSTER_ID"
  DELTA_JS_LINES_CONSTRAINT="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["_PHASE_LC_ROOT"]) / "projects" / os.environ["_PHASE_LC_PROJECT"] / "systems" / os.environ["_PHASE_LC_SYSTEM"]
cluster = root / "clusters" / os.environ["_PHASE_LC_CLUSTER"]
analyses = cluster / "analyses" / "delta_js"
if not analyses.exists():
    raise SystemExit(0)
for d in sorted([p for p in analyses.iterdir() if p.is_dir()]):
    aid = d.name
    meta_path = d / "analysis_metadata.json"
    label = aid
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            created = str(m.get("created_at") or "")[:19].replace("T", " ")
            a = str(m.get("model_a_name") or m.get("model_a_id") or "A")
            b = str(m.get("model_b_name") or m.get("model_b_id") or "B")
            label = f"{created} :: {a} vs {b}"
        except Exception:
            pass
    print(f"{aid}|{label}")
PY
)"
  unset _PHASE_LC_ROOT _PHASE_LC_PROJECT _PHASE_LC_SYSTEM _PHASE_LC_CLUSTER
  if [ -z "$(trim "$DELTA_JS_LINES_CONSTRAINT")" ]; then
    echo "No delta_js analyses found in this cluster. Use manual constraint mode or run Delta JS first."
    exit 1
  fi
  DJS_CONSTRAINT_ROW="$(offline_choose_one "Select Delta JS analysis for auto constraints/success:" "$DELTA_JS_LINES_CONSTRAINT")"
  DELTA_JS_EXPERIMENT_ID="$(printf "%s" "$DJS_CONSTRAINT_ROW" | awk -F'|' '{print $1}')"
  if [ -z "$DELTA_JS_EXPERIMENT_ID" ]; then
    echo "Delta JS analysis selection is required in delta_js_auto mode."
    exit 1
  fi
  SAMPLE_ROW_AUTO="$(offline_choose_one "Select sample for ligand-specific impact (blank uses start MD):" "$MD_LINES")"
  CONSTRAINT_DELTA_JS_SAMPLE_ID="$(printf "%s" "$SAMPLE_ROW_AUTO" | awk -F'|' '{print $1}')"
  if [ -z "$CONSTRAINT_DELTA_JS_SAMPLE_ID" ]; then
    CONSTRAINT_DELTA_JS_SAMPLE_ID="$START_SAMPLE_ID"
  fi
  CONSTRAINT_AUTO_TOPK="$(prompt "Auto constraints: top-K residues" "12")"
  CONSTRAINT_AUTO_EDGE_ALPHA="$(prompt "Auto constraints: edge blend alpha [0..1]" "0.3")"
  CONSTRAINT_AUTO_EXCLUDE_SUCCESS="$(prompt "Auto constraints: exclude success residues? (Y/n)" "Y")"
fi

SAMPLER="$(prompt "Sampler (sa/gibbs)" "sa")"
SAMPLER="$(printf "%s" "$SAMPLER" | tr '[:upper:]' '[:lower:]')"
if [ "$SAMPLER" != "gibbs" ]; then
  SAMPLER="sa"
fi

LAMBDAS="$(prompt "Lambda grid" "0,0.25,0.5,1,2,4,8")"
N_START="$(prompt "Random MD frames" "100")"
N_PER_FRAME="$(prompt "Samples per frame (tail subsample)" "100")"
N_STEPS="$(prompt "Sampling steps per trajectory" "1000")"
TAIL_STEPS="$(prompt "Tail steps used for JS/success" "200")"

WINDOW_SIZE="$(prompt "Target window size" "11")"
PSEUDOCOUNT="$(prompt "Target pseudocount" "1e-3")"
EPS_LOG="$(prompt "Log-penalty epsilon" "1e-8")"

WEIGHT_MODE="uniform"
WEIGHTS=""
if [ "$CONSTRAINT_MODE" = "manual" ]; then
  WEIGHT_MODE="$(prompt "Constraint weight mode (uniform/js_abs/custom)" "uniform")"
  WEIGHT_MODE="$(printf "%s" "$WEIGHT_MODE" | tr '[:upper:]' '[:lower:]')"
  if [ "$WEIGHT_MODE" != "js_abs" ] && [ "$WEIGHT_MODE" != "custom" ]; then
    WEIGHT_MODE="uniform"
  fi
  if [ "$WEIGHT_MODE" = "custom" ]; then
    WEIGHTS="$(prompt "Constraint weights (comma-separated)" "")"
    WEIGHTS="$(trim "$WEIGHTS")"
  fi
else
  echo "Constraint weights will be derived automatically from Delta-JS impact."
fi
W_MIN="$(prompt "Constraint weight min" "0.0")"
W_MAX="$(prompt "Constraint weight max" "1.0")"

GIBBS_BETA="$(prompt "Gibbs beta" "1.0")"
SA_HOT="$(prompt "SA beta hot" "0.8")"
SA_COLD="$(prompt "SA beta cold" "50.0")"
SA_SCHEDULE="$(prompt "SA schedule (geom/lin)" "geom")"

LABEL_MODE="$(prompt "MD label mode (assigned/halo)" "assigned")"
LABEL_MODE="$(printf "%s" "$LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$LABEL_MODE" != "halo" ]; then
  LABEL_MODE="assigned"
fi

SUCCESS_MODE="$(prompt "Success metric (deltae/delta_js_edge)" "deltae")"
SUCCESS_MODE="$(printf "%s" "$SUCCESS_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$SUCCESS_MODE" != "delta_js_edge" ]; then
  SUCCESS_MODE="deltae"
fi
DELTA_JS_D_RES_MIN="0.0"
DELTA_JS_D_RES_MAX=""
DELTA_JS_D_EDGE_MIN="0.0"
DELTA_JS_D_EDGE_MAX=""
DELTA_JS_ALPHA=""
JS_SUCCESS_THRESHOLD="0.10"
JS_SUCCESS_MARGIN="0.0"
if [ "$SUCCESS_MODE" = "delta_js_edge" ]; then
  if [ -z "$DELTA_JS_EXPERIMENT_ID" ]; then
    export _PHASE_LC_ROOT="$OFFLINE_ROOT"
    export _PHASE_LC_PROJECT="$OFFLINE_PROJECT_ID"
    export _PHASE_LC_SYSTEM="$OFFLINE_SYSTEM_ID"
    export _PHASE_LC_CLUSTER="$CLUSTER_ID"
    DELTA_JS_LINES="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["_PHASE_LC_ROOT"]) / "projects" / os.environ["_PHASE_LC_PROJECT"] / "systems" / os.environ["_PHASE_LC_SYSTEM"]
cluster = root / "clusters" / os.environ["_PHASE_LC_CLUSTER"]
analyses = cluster / "analyses" / "delta_js"
if not analyses.exists():
    raise SystemExit(0)
for d in sorted([p for p in analyses.iterdir() if p.is_dir()]):
    aid = d.name
    meta_path = d / "analysis_metadata.json"
    label = aid
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            created = str(m.get("created_at") or "")[:19].replace("T", " ")
            a = str(m.get("model_a_name") or m.get("model_a_id") or "A")
            b = str(m.get("model_b_name") or m.get("model_b_id") or "B")
            label = f"{created} :: {a} vs {b}"
        except Exception:
            pass
    print(f"{aid}|{label}")
PY
)"
    unset _PHASE_LC_ROOT _PHASE_LC_PROJECT _PHASE_LC_SYSTEM _PHASE_LC_CLUSTER
    if [ -z "$(trim "$DELTA_JS_LINES")" ]; then
      echo "No delta_js analyses found in this cluster. Use success metric deltae or run Delta JS first."
      exit 1
    fi
    DELTA_JS_ROW="$(offline_choose_one "Select Delta JS analysis (shared experiment):" "$DELTA_JS_LINES")"
    DELTA_JS_EXPERIMENT_ID="$(printf "%s" "$DELTA_JS_ROW" | awk -F'|' '{print $1}')"
    if [ -z "$DELTA_JS_EXPERIMENT_ID" ]; then
      echo "Delta JS analysis selection is required for success mode delta_js_edge."
      exit 1
    fi
  fi
  DELTA_JS_D_RES_MIN="$(prompt "Delta-JS residue D min filter" "0.0")"
  DELTA_JS_D_RES_MAX="$(prompt "Delta-JS residue D max filter (blank=none)" "")"
  DELTA_JS_D_EDGE_MIN="$(prompt "Delta-JS edge D min filter" "0.0")"
  DELTA_JS_D_EDGE_MAX="$(prompt "Delta-JS edge D max filter (blank=none)" "")"
  DELTA_JS_ALPHA="$(prompt "Delta-JS node-edge alpha (blank=analysis default)" "")"
  JS_SUCCESS_THRESHOLD="$(prompt "JS success threshold" "0.10")"
  JS_SUCCESS_MARGIN="$(prompt "JS A/B margin" "0.0")"
fi

if [ "$CONSTRAINT_MODE" = "delta_js_auto" ] || [ "$SUCCESS_MODE" = "delta_js_edge" ]; then
  export _PHASE_LC_ROOT="$OFFLINE_ROOT"
  export _PHASE_LC_PROJECT="$OFFLINE_PROJECT_ID"
  export _PHASE_LC_SYSTEM="$OFFLINE_SYSTEM_ID"
  export _PHASE_LC_CLUSTER="$CLUSTER_ID"
  FILTER_SETUP_LINES="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["_PHASE_LC_ROOT"]) / "projects" / os.environ["_PHASE_LC_PROJECT"] / "systems" / os.environ["_PHASE_LC_SYSTEM"]
cluster = root / "clusters" / os.environ["_PHASE_LC_CLUSTER"]
ui_dir = cluster / "ui_setups"
if not ui_dir.exists():
    raise SystemExit(0)
for p in sorted(ui_dir.glob("*.json")):
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if str(obj.get("setup_type") or "") != "js_range_filters":
        continue
    if str(obj.get("page") or "") != "delta_js":
        continue
    sid = str(obj.get("setup_id") or p.stem)
    name = str(obj.get("name") or sid)
    print(f"{sid}|{name}")
PY
)"
  unset _PHASE_LC_ROOT _PHASE_LC_PROJECT _PHASE_LC_SYSTEM _PHASE_LC_CLUSTER
  if [ -n "$(trim "$FILTER_SETUP_LINES")" ]; then
    FILTER_ROW="$(offline_choose_one "Optional Delta-JS filter setup (blank=none):" "$FILTER_SETUP_LINES")"
    DELTA_JS_FILTER_SETUP_ID="$(printf "%s" "$FILTER_ROW" | awk -F'|' '{print $1}')"
  fi
  DELTA_JS_FILTER_EDGE_ALPHA="$(prompt "Delta-JS filter edge alpha [0..1]" "0.75")"
fi

DELTAE_MARGIN="$(prompt "DeltaE success margin" "0.0")"
P_TARGET="$(prompt "Completion target success (for cost)" "0.7")"
COST_UNREACHED="$(prompt "Completion cost if unreached (blank=auto)" "")"
SEED="$(prompt "Random seed" "0")"
WORKERS="$(prompt "Worker processes (0=all CPUs)" "0")"
KEEP_INVALID="$(prompt "Keep invalid frames? (y/N)" "N")"
SHOW_PROGRESS="$(prompt "Show progress output? (Y/n)" "Y")"

if ! [[ "$WORKERS" =~ ^-?[0-9]+$ ]]; then
  WORKERS="1"
fi
if [ "$WORKERS" -le 0 ]; then
  WORKERS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
if [ "$WORKERS" -le 0 ]; then
  WORKERS="1"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_ligand_completion
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-a-id "$MODEL_A_ID"
  --model-b-id "$MODEL_B_ID"
  --md-sample-id "$START_SAMPLE_ID"
  --constrained-residues "$CONSTRAINED"
  --sampler "$SAMPLER"
  --lambda-values "$LAMBDAS"
  --n-start-frames "$N_START"
  --n-samples-per-frame "$N_PER_FRAME"
  --n-steps "$N_STEPS"
  --tail-steps "$TAIL_STEPS"
  --target-window-size "$WINDOW_SIZE"
  --target-pseudocount "$PSEUDOCOUNT"
  --epsilon-logpenalty "$EPS_LOG"
  --constraint-source-mode "$CONSTRAINT_MODE"
  --constraint-weight-mode "$WEIGHT_MODE"
  --constraint-weight-min "$W_MIN"
  --constraint-weight-max "$W_MAX"
  --gibbs-beta "$GIBBS_BETA"
  --sa-beta-hot "$SA_HOT"
  --sa-beta-cold "$SA_COLD"
  --sa-schedule "$SA_SCHEDULE"
  --md-label-mode "$LABEL_MODE"
  --success-metric-mode "$SUCCESS_MODE"
  --deltae-margin "$DELTAE_MARGIN"
  --completion-target-success "$P_TARGET"
  --workers "$WORKERS"
  --seed "$SEED"
)

if [ -n "$REF_A_ID" ]; then
  CMD+=(--reference-sample-id-a "$REF_A_ID")
fi
if [ -n "$REF_B_ID" ]; then
  CMD+=(--reference-sample-id-b "$REF_B_ID")
fi
if [ "$WEIGHT_MODE" = "custom" ] && [ -n "$WEIGHTS" ]; then
  CMD+=(--constraint-weights "$WEIGHTS")
fi
if [ "$CONSTRAINT_MODE" = "delta_js_auto" ]; then
  CMD+=(--constraint-delta-js-analysis-id "$DELTA_JS_EXPERIMENT_ID")
  CMD+=(--constraint-delta-js-sample-id "$CONSTRAINT_DELTA_JS_SAMPLE_ID")
  CMD+=(--constraint-auto-top-k "$CONSTRAINT_AUTO_TOPK")
  CMD+=(--constraint-auto-edge-alpha "$CONSTRAINT_AUTO_EDGE_ALPHA")
  if [ "$CONSTRAINT_AUTO_EXCLUDE_SUCCESS" = "n" ] || [ "$CONSTRAINT_AUTO_EXCLUDE_SUCCESS" = "no" ]; then
    CMD+=(--no-constraint-auto-exclude-success)
  else
    CMD+=(--constraint-auto-exclude-success)
  fi
fi
if [ "$SUCCESS_MODE" = "delta_js_edge" ]; then
  CMD+=(--delta-js-analysis-id "$DELTA_JS_EXPERIMENT_ID")
  CMD+=(--delta-js-d-residue-min "$DELTA_JS_D_RES_MIN")
  CMD+=(--delta-js-d-edge-min "$DELTA_JS_D_EDGE_MIN")
  CMD+=(--js-success-threshold "$JS_SUCCESS_THRESHOLD")
  CMD+=(--js-success-margin "$JS_SUCCESS_MARGIN")
  if [ -n "$(trim "$DELTA_JS_D_RES_MAX")" ]; then
    CMD+=(--delta-js-d-residue-max "$DELTA_JS_D_RES_MAX")
  fi
  if [ -n "$(trim "$DELTA_JS_D_EDGE_MAX")" ]; then
    CMD+=(--delta-js-d-edge-max "$DELTA_JS_D_EDGE_MAX")
  fi
  if [ -n "$(trim "$DELTA_JS_ALPHA")" ]; then
    CMD+=(--delta-js-node-edge-alpha "$DELTA_JS_ALPHA")
  fi
fi
if [ -n "$(trim "$DELTA_JS_EXPERIMENT_ID")" ]; then
  CMD+=(--delta-js-experiment-id "$DELTA_JS_EXPERIMENT_ID")
fi
if [ -n "$(trim "$DELTA_JS_FILTER_SETUP_ID")" ]; then
  CMD+=(--delta-js-filter-setup-id "$DELTA_JS_FILTER_SETUP_ID")
fi
CMD+=(--delta-js-filter-edge-alpha "$DELTA_JS_FILTER_EDGE_ALPHA")
if [ -n "$(trim "$COST_UNREACHED")" ]; then
  CMD+=(--completion-cost-if-unreached "$COST_UNREACHED")
fi
if [ "$KEEP_INVALID" = "y" ] || [ "$KEEP_INVALID" = "yes" ] || [ "$KEEP_INVALID" = "true" ]; then
  CMD+=(--keep-invalid)
fi
if [ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ]; then
  CMD+=(--progress)
fi

echo ""
echo "Running ligand completion analysis..."
echo "  cluster: $CLUSTER_ID"
echo "  model A: ${MODEL_A_NAME:-$MODEL_A_ID} ($MODEL_A_ID)"
echo "  model B: ${MODEL_B_NAME:-$MODEL_B_ID} ($MODEL_B_ID)"
echo "  start MD: ${START_SAMPLE_NAME:-$START_SAMPLE_ID} ($START_SAMPLE_ID)"
echo ""

exec "${CMD[@]}"
