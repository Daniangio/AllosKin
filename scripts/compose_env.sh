#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"
DATA_ROOT_VAL="${PHASE_DATA_ROOT:-}"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -f "$ENV_FILE" ]; then
  # Remove previous values (keep any other compose env vars intact).
  grep -v -E '^(PHASE_UID|PHASE_GID|PHASE_DATA_ROOT)=' "$ENV_FILE" > "$tmp" || true
else
  : > "$tmp"
fi

{
  echo "PHASE_UID=${UID_VAL}"
  echo "PHASE_GID=${GID_VAL}"
  if [ -n "$DATA_ROOT_VAL" ]; then
    echo "PHASE_DATA_ROOT=${DATA_ROOT_VAL}"
  fi
} >> "$tmp"

mv "$tmp" "$ENV_FILE"
trap - EXIT

echo "Wrote ${ENV_FILE}"
echo "  PHASE_UID=${UID_VAL}"
echo "  PHASE_GID=${GID_VAL}"
if [ -n "$DATA_ROOT_VAL" ]; then
  echo "  PHASE_DATA_ROOT=${DATA_ROOT_VAL}"
else
  echo "  PHASE_DATA_ROOT not exported; docker-compose will fall back to ./data"
fi
