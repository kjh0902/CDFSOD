#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] Activate the conda environment created from environment.yml first." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMCV_COMMIT="a8073c74bf83d62ec36a103f835faa4837fb6585"
MMCV_SOURCE_DIR="${MMCV_SOURCE_DIR:-${CONDA_PREFIX}/src/mmcv-${MMCV_COMMIT:0:12}}"

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="12.0"
export MMCV_WITH_OPS=1
export MAX_JOBS="${MAX_JOBS:-2}"

if [[ ! -x "${CUDACXX}" ]]; then
  echo "[ERROR] CUDA 12.8 nvcc was not found at ${CUDACXX}." >&2
  exit 1
fi

python -m pip install --upgrade \
  "pip==26.2.1" "setuptools==81.0.0" "wheel==0.45.1" "ninja==1.13.0"

python -m pip install \
  "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip install --no-deps "mmengine==0.10.7"

mkdir -p "$(dirname "${MMCV_SOURCE_DIR}")"
if [[ ! -d "${MMCV_SOURCE_DIR}/.git" ]]; then
  git clone https://github.com/open-mmlab/mmcv.git "${MMCV_SOURCE_DIR}"
fi

MMCV_CURRENT_COMMIT="$(git -C "${MMCV_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${MMCV_CURRENT_COMMIT}" != "${MMCV_COMMIT}" ]]; then
  git -C "${MMCV_SOURCE_DIR}" fetch origin "${MMCV_COMMIT}"
  git -C "${MMCV_SOURCE_DIR}" checkout --detach "${MMCV_COMMIT}"
fi

python -m pip install \
  --no-build-isolation --no-deps --force-reinstall -v "${MMCV_SOURCE_DIR}"

python "${REPO_ROOT}/tools/patch_mmengine_determinism.py"
CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/tools/verify_environment.py"

echo "[OK] FT-FSOD RTX 5090 environment is ready."
