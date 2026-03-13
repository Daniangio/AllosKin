#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${PHASE_DATA_ROOT:-${ROOT_DIR}/data}}"
TARGET_UID="${TARGET_UID:-${SUDO_UID:-}}"
TARGET_GID="${TARGET_GID:-${SUDO_GID:-}}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script with sudo so it can fix ownership under ${DATA_ROOT}." >&2
  exit 1
fi

if [ -z "${TARGET_UID}" ] || [ -z "${TARGET_GID}" ]; then
  echo "TARGET_UID/TARGET_GID not set and sudo did not provide SUDO_UID/SUDO_GID." >&2
  echo "Example: sudo TARGET_UID=$(id -u) TARGET_GID=$(id -g) $0 ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}"

echo "Fixing ownership under ${DATA_ROOT}"
echo "Target owner: ${TARGET_UID}:${TARGET_GID}"

chown -R "${TARGET_UID}:${TARGET_GID}" "${DATA_ROOT}"
chmod -R u+rwX,g+rwX,o+rX "${DATA_ROOT}"

echo "Done."
