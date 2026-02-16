#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

DATA_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"
PROJECT_ID="${OFFLINE_PROJECT_ID:-}"
SYSTEM_ID="${OFFLINE_SYSTEM_ID:-}"
CLUSTER_ID="${OFFLINE_CLUSTER_ID:-}"
ACTION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --root) DATA_ROOT="$2"; shift 2 ;;
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --system-id) SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id) CLUSTER_ID="$2"; shift 2 ;;
    --action) ACTION="$2"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/cluster_patch.sh [--root DATA_ROOT] --project-id ID --system-id ID --cluster-id ID [--action ACTION]

Interactive actions:
  list      List preview patches
  create    Create preview patch on selected residues
  confirm   Confirm preview patch and recompute MD memberships
  discard   Discard preview patch
USAGE
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$PROJECT_ID" ] || [ -z "$SYSTEM_ID" ] || [ -z "$CLUSTER_ID" ]; then
  echo "Missing --project-id/--system-id/--cluster-id" >&2
  exit 1
fi

export PHASE_DATA_ROOT="$DATA_ROOT"

choose_patch() {
  local mode="${1:-all}"
  local rows
  rows="$(python -m phase.scripts.cluster_patch \
    --root "$DATA_ROOT" \
    --project-id "$PROJECT_ID" \
    --system-id "$SYSTEM_ID" \
    --cluster-id "$CLUSTER_ID" \
    list --pipe || true)"
  if [ "$mode" = "preview" ]; then
    rows="$(printf "%s\n" "$rows" | awk -F'|' '$3=="preview"')"
  fi
  if [ -z "${rows:-}" ]; then
    echo ""
    return 0
  fi
  local selected
  selected="$(offline_choose_one "Available patches:" "$rows")"
  printf "%s" "$selected" | awk -F'|' '{print $1}'
}

run_action() {
  local action="$1"
  case "$action" in
    list)
      python -m phase.scripts.cluster_patch \
        --root "$DATA_ROOT" \
        --project-id "$PROJECT_ID" \
        --system-id "$SYSTEM_ID" \
        --cluster-id "$CLUSTER_ID" \
        list
      ;;
    create)
      local residues
      residues="$(prompt "Residue keys (comma-separated, e.g. res_120,res_121)" "")"
      residues="$(trim "$residues")"
      if [ -z "$residues" ]; then
        echo "Residue keys are required."
        return 1
      fi
      local mode n_clusters inconsistent_threshold inconsistent_depth max_cluster_frames linkage covariance halo_pct patch_name
      mode="$(prompt "Cluster selection mode (maxclust/inconsistent)" "maxclust")"
      mode="$(printf "%s" "$mode" | tr '[:upper:]' '[:lower:]')"
      n_clusters=""
      inconsistent_threshold=""
      inconsistent_depth="2"
      if [ "$mode" = "inconsistent" ]; then
        inconsistent_threshold="$(prompt "Inconsistent threshold t" "1.0")"
        inconsistent_depth="$(prompt "Inconsistent depth" "2")"
      else
        mode="maxclust"
        n_clusters="$(prompt "n_clusters (blank = keep current per-residue K)" "")"
      fi
      max_cluster_frames="$(prompt "Max frames for patch clustering (blank = all frames)" "")"
      linkage="$(prompt "Linkage method (ward/complete/average/single)" "ward")"
      covariance="$(prompt "Covariance type (full/diag)" "full")"
      halo_pct="$(prompt "Halo percentile" "5.0")"
      patch_name="$(prompt "Patch name (optional)" "")"

      cmd=(
        python -m phase.scripts.cluster_patch
        --root "$DATA_ROOT"
        --project-id "$PROJECT_ID"
        --system-id "$SYSTEM_ID"
        --cluster-id "$CLUSTER_ID"
        create
        --residue-keys "$residues"
        --cluster-selection-mode "$mode"
        --linkage-method "$linkage"
        --covariance-type "$covariance"
        --halo-percentile "$halo_pct"
      )
      if [ -n "$n_clusters" ]; then
        cmd+=(--n-clusters "$n_clusters")
      fi
      if [ "$mode" = "inconsistent" ]; then
        cmd+=(--inconsistent-threshold "$inconsistent_threshold" --inconsistent-depth "$inconsistent_depth")
      fi
      if [ -n "$max_cluster_frames" ]; then
        cmd+=(--max-cluster-frames "$max_cluster_frames")
      fi
      if [ -n "$patch_name" ]; then
        cmd+=(--name "$patch_name")
      fi
      "${cmd[@]}"
      ;;
    confirm)
      local patch_id
      patch_id="$(choose_patch preview)"
      if [ -z "$patch_id" ]; then
        echo "No preview patch selected."
        return 0
      fi
      python -m phase.scripts.cluster_patch \
        --root "$DATA_ROOT" \
        --project-id "$PROJECT_ID" \
        --system-id "$SYSTEM_ID" \
        --cluster-id "$CLUSTER_ID" \
        confirm \
        --patch-id "$patch_id"
      ;;
    discard)
      local patch_id
      patch_id="$(choose_patch all)"
      if [ -z "$patch_id" ]; then
        echo "No patch selected."
        return 0
      fi
      python -m phase.scripts.cluster_patch \
        --root "$DATA_ROOT" \
        --project-id "$PROJECT_ID" \
        --system-id "$SYSTEM_ID" \
        --cluster-id "$CLUSTER_ID" \
        discard \
        --patch-id "$patch_id"
      ;;
    *)
      echo "Unsupported action: $action" >&2
      return 1
      ;;
  esac
}

if [ -n "$ACTION" ]; then
  run_action "$ACTION"
  exit 0
fi

menu_lines=$'list|List preview patches\ncreate|Create residue patch preview\nconfirm|Confirm preview patch\ndiscard|Discard preview patch\nback|Back'
selected="$(offline_choose_one "Cluster patch actions:" "$menu_lines")"
sel_action="$(printf "%s" "$selected" | awk -F'|' '{print $1}')"
if [ -z "$sel_action" ] || [ "$sel_action" = "back" ]; then
  exit 0
fi
run_action "$sel_action"
