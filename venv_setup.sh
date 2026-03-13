#!/usr/bin/env bash
set -euo pipefail

# --- 1. SANITIZATION ---
# Unset variables that cause "leakage" from Conda or system paths
unset PYTHONPATH
unset LD_LIBRARY_PATH
# -----------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PY="3.10"
DEFAULT_DEPS_DIR="${ROOT_DIR}/deps"
DEFAULT_GEQTRAIN_REF="main"
DEFAULT_GEQDIFF_REF="main"

# Default venv location logic
if [ -d "${ROOT_DIR}/.venv-phase" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-phase"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-phase"
fi

prompt() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  if [ -z "$var" ]; then echo "$default"; else echo "$var"; fi
}

sync_repo() {
  local url="$1"
  local dir="$2"
  local ref="$3"
  local update_existing="$4"

  mkdir -p "$(dirname "${dir}")"

  if [[ ! -d "${dir}/.git" ]]; then
    echo "Cloning ${url} -> ${dir}"
    git clone "${url}" "${dir}"
  else
    echo "Repository already exists: ${dir}"
    if [[ "${update_existing}" =~ ^[Yy]$ ]]; then
      echo "Fetching updates for ${dir}"
      git -C "${dir}" fetch --all --tags
    fi
  fi

  if [[ -z "${ref}" ]]; then
    return
  fi

  if git -C "${dir}" show-ref --verify --quiet "refs/heads/${ref}"; then
    git -C "${dir}" checkout "${ref}"
    if [[ "${update_existing}" =~ ^[Yy]$ ]]; then
      git -C "${dir}" pull --ff-only || true
    fi
    return
  fi

  if git -C "${dir}" show-ref --verify --quiet "refs/tags/${ref}"; then
    git -C "${dir}" checkout "tags/${ref}"
    return
  fi

  if git -C "${dir}" ls-remote --exit-code --heads origin "${ref}" >/dev/null 2>&1; then
    git -C "${dir}" fetch origin "${ref}"
    git -C "${dir}" checkout -B "${ref}" "origin/${ref}"
    return
  fi

  git -C "${dir}" checkout "${ref}"
}

install_rdkit() {
  if uv pip install rdkit; then
    return
  fi

  echo "PyPI install for rdkit failed; trying rdkit-pypi..."
  uv pip install rdkit-pypi
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

INSTALL_BACKMAPPING_STACK="$(prompt "Install backmapping stack (GEqTrain + GEqDiff) now (y/N)" "Y")"
if [[ "$INSTALL_BACKMAPPING_STACK" =~ ^[Yy]$ ]]; then
  DEPS_DIR="$(prompt "Dependencies directory" "${DEFAULT_DEPS_DIR}")"
  GEQTRAIN_REF="$(prompt "GEqTrain git ref" "${DEFAULT_GEQTRAIN_REF}")"
  GEQDIFF_REF="$(prompt "GEqDiff git ref" "${DEFAULT_GEQDIFF_REF}")"
  UPDATE_REPOS="$(prompt "Update existing GEqTrain/GEqDiff repos (y/N)" "N")"

  GEQTRAIN_DIR="${DEPS_DIR}/GEqTrain"
  GEQDIFF_DIR="${DEPS_DIR}/GEqDiff"

  sync_repo "https://github.com/limresgrp/GEqTrain.git" "${GEQTRAIN_DIR}" "${GEQTRAIN_REF}" "${UPDATE_REPOS}"
  sync_repo "https://github.com/limresgrp/GEqDiff.git" "${GEQDIFF_DIR}" "${GEQDIFF_REF}" "${UPDATE_REPOS}"

  echo "Installing GEqTrain/GEqDiff runtime dependencies..."
  BACKMAPPING_DEPS=(
    numpy
    scipy
    tqdm
    e3nn
    MDAnalysis
    matplotlib
    plotly
  )
  uv pip install "${BACKMAPPING_DEPS[@]}"
  install_rdkit

  echo "Installing GEqTrain (editable)..."
  uv pip install -e "${GEQTRAIN_DIR}"

  echo "Installing GEqDiff (editable)..."
  uv pip install -e "${GEQDIFF_DIR}"
fi

echo "Installing phase package (no deps)..."
uv pip install -e "${ROOT_DIR}" --no-deps

echo "---------------------------------------------------"
echo "Build Complete. Activate your environment with:"
echo "source ${VENV_DIR}/bin/activate"
if [[ "${INSTALL_BACKMAPPING_STACK}" =~ ^[Yy]$ ]]; then
  echo "Backmapping repos:"
  echo "  GEqTrain: ${GEQTRAIN_DIR}"
  echo "  GEqDiff:  ${GEQDIFF_DIR}"
fi
echo "---------------------------------------------------"
