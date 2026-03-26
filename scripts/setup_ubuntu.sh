#!/usr/bin/env bash
set -euo pipefail

# Lightweight installer using micromamba or conda
# Usage:
#   ./scripts/setup_ubuntu.sh
# Installs env named "gwsearch" with conda-forge packages.

ENV_NAME="gwsearch"
PY_VERSION="3.10"
MAMBA_ROOT="${HOME}/.local/micromamba"
MICROMAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"

# Detect existing conda/mamba
if command -v micromamba >/dev/null 2>&1; then
  MICROMAMBA_BIN="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
  MICROMAMBA_BIN="$(command -v mamba)"
elif [ ! -x "${MICROMAMBA_BIN}" ]; then
  echo "Installing micromamba to ${MAMBA_ROOT}..."
  mkdir -p "${MAMBA_ROOT}"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "${MAMBA_ROOT}" --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
"${MICROMAMBA_BIN}" shell init -s bash >/dev/null 2>&1 || true
# shellcheck disable=SC1090
source "${HOME}/.bashrc" 2>/dev/null || true

echo "Creating/updating environment ${ENV_NAME}..."
"${MICROMAMBA_BIN}" create -y -n "${ENV_NAME}" -f environment.yml || \
"${MICROMAMBA_BIN}" update -y -n "${ENV_NAME}" -f environment.yml

echo "Activating environment..."
# shellcheck disable=SC1091
source "${MAMBA_ROOT}/etc/profile.d/conda.sh" 2>/dev/null || true
# shellcheck disable=SC1091
source "${MAMBA_ROOT}/etc/profile.d/micromamba.sh" 2>/dev/null || true
conda activate "${ENV_NAME}" 2>/dev/null || micromamba activate "${ENV_NAME}"

echo "Installing gwsearch in editable mode..."
pip install -e .

echo "Done. To activate later: micromamba activate ${ENV_NAME}"