#!/usr/bin/env bash
set -euo pipefail

read -r -p "Reads [500]: " READS
READS=${READS:-500}
read -r -p "Sweeps list (comma) [10,50,200,1000]: " SWEEPS
SWEEPS=${SWEEPS:-10,50,200,1000}
read -r -p "Beta hot list (comma, pairs with cold list) [0.1,0.5]: " BETA_HOT
BETA_HOT=${BETA_HOT:-0.1,0.5}
read -r -p "Beta cold list (comma, pairs with hot list) [2,5]: " BETA_COLD
BETA_COLD=${BETA_COLD:-2,5}
read -r -p "Restart top-k [50]: " TOPK
TOPK=${TOPK:-50}
read -r -p "CSV path [scripts/debug_sa_results.csv]: " CSV_PATH
CSV_PATH=${CSV_PATH:-scripts/debug_sa_results.csv}
read -r -p "Also run auto schedule? (y/n) [y]: " ALSO_AUTO
ALSO_AUTO=${ALSO_AUTO:-y}

IFS=',' read -r -a SWEEP_ARR <<< "$SWEEPS"
IFS=',' read -r -a HOT_ARR <<< "$BETA_HOT"
IFS=',' read -r -a COLD_ARR <<< "$BETA_COLD"

for sweeps in "${SWEEP_ARR[@]}"; do
  EXTRA_ARGS=(--beta-hot-list "$BETA_HOT" --beta-cold-list "$BETA_COLD")
  if [[ "$ALSO_AUTO" == "y" || "$ALSO_AUTO" == "Y" ]]; then
    EXTRA_ARGS+=(--also-auto)
  fi
  echo "== sweeps=$sweeps reads=$READS hot_list=$BETA_HOT cold_list=$BETA_COLD =="
  python scripts/debug_sa_init.py --reads "$READS" --sweeps "$sweeps" --restart-topk "$TOPK" --csv "$CSV_PATH" "${EXTRA_ARGS[@]}"
done

read -r -p "Analyze CSV and plot now? (y/n) [y]: " ANALYZE
ANALYZE=${ANALYZE:-y}
if [[ "$ANALYZE" == "y" || "$ANALYZE" == "Y" ]]; then
  python scripts/debug_sa_init.py --csv "$CSV_PATH" --analyze
fi
