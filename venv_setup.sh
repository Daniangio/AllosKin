#!/usr/bin/env bash
set -euo pipefail

# --- 1. SANITIZATION ---
# Unset variables that cause "leakage" from Conda or system paths
unset PYTHONPATH
unset LD_LIBRARY_PATH
# -----------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PY="3.10"

# Default venv location logic
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
  if [ -z "$var" ]; then echo "$default"; else echo "$var"; fi
}

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

PY_VER="$(prompt "Python version" "${DEFAULT_PY}")"
VENV_DIR="$(prompt "Virtual env directory" "${DEFAULT_ENV}")"

# --- 2. CLEAN INSTALL LOGIC ---
if [ -d "${VENV_DIR}" ]; then
  echo "Existing environment found at: ${VENV_DIR}"
  echo "Removing it to ensure a clean build..."
  rm -rf "${VENV_DIR}"
fi

echo "Creating fresh isolated venv with Python ${PY_VER}..."
# --seed ensures pip/setuptools are fresh
if ! uv venv "${VENV_DIR}" --python "${PY_VER}" --seed; then
    echo "Local python version not found. Downloading via uv..."
    uv python install "${PY_VER}"
    uv venv "${VENV_DIR}" --python "${PY_VER}" --seed
fi
# ------------------------------

source "${VENV_DIR}/bin/activate"

# Verify we are using the correct python
CURRENT_PYTHON=$(which python)
if [[ "$CURRENT_PYTHON" != *"${VENV_DIR}"* ]]; then
    echo "CRITICAL ERROR: Failed to activate venv. Current python: $CURRENT_PYTHON"
    exit 1
fi

REQ_TMP="$(mktemp)"
CONSTRAINTS_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP" "$CONSTRAINTS_TMP"' EXIT

# Filter out torch from main requirements to handle it separately
grep -v -E '^(torch)($|[<>=])' "${ROOT_DIR}/requirements.txt" > "$REQ_TMP"
echo "numpy<2" > "$CONSTRAINTS_TMP"

echo "Installing base dependencies (numpy, tqdm)..."
uv pip install -r "$CONSTRAINTS_TMP"
uv pip install tqdm

echo "Installing full dependencies (excluding torch)..."
uv pip install -r "$REQ_TMP" --constraints "$CONSTRAINTS_TMP"

INSTALL_TORCH="$(prompt "Install torch now (y/N)" "Y")"
if [[ "$INSTALL_TORCH" =~ ^[Yy]$ ]]; then
  CUDA_DEFAULT="cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DEFAULT="cu128"
  fi
  CUDA_FLAVOR="$(prompt "Torch CUDA flavor (cpu/cu118/cu121/cu124/cu128)" "${CUDA_DEFAULT}")"
  TORCH_VERSION="$(prompt "Torch version (blank = latest)" "")"
  TORCH_SPEC="torch"
  if [ -n "$TORCH_VERSION" ]; then
    TORCH_SPEC="torch==${TORCH_VERSION}"
  fi
  echo "Installing torch (${CUDA_FLAVOR})..."
  uv pip install "$TORCH_SPEC" --torch-backend "${CUDA_FLAVOR}"
fi

echo "Installing phase package (no deps)..."
uv pip install -e "${ROOT_DIR}" --no-deps

echo "---------------------------------------------------"
echo "Build Complete. Activate your environment with:"
echo "source ${VENV_DIR}/bin/activate"
echo "---------------------------------------------------"