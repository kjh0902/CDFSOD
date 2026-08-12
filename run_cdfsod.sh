#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one CD-FSOD dataset/shot experiment (train, then evaluate the best checkpoint).

Usage:
  bash run_cdfsod.sh --dataset DATASET --shot {1|5|10} [options]

Required:
  --dataset NAME   ArTaxOr, Clipart1k, DIOR, FISH (or DeepFish), NEU-DET, UODD
  --shot N         1, 5, or 10

Options:
  --gpu ID         Physical GPU ID (default: 0; this repository supports GPU 0 only)
  --port PORT      torch.distributed rendezvous port (default: 29500)
  --resume         Resume from the latest checkpoint in the experiment directory
  --dry-run        Validate arguments and print resolved paths without running
  -h, --help       Show this help
EOF
}

DATASET=""
SHOT=""
GPU_ID="0"
PORT="29500"
RESUME=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "[ERROR] --dataset requires a value." >&2; exit 2; }
      DATASET="$2"
      shift 2
      ;;
    --shot)
      [[ $# -ge 2 ]] || { echo "[ERROR] --shot requires a value." >&2; exit 2; }
      SHOT="$2"
      shift 2
      ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "[ERROR] --gpu requires a value." >&2; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "[ERROR] --port requires a value." >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${DATASET}" ]] || { echo "[ERROR] --dataset is required." >&2; usage >&2; exit 2; }
[[ -n "${SHOT}" ]] || { echo "[ERROR] --shot is required." >&2; usage >&2; exit 2; }

case "${DATASET,,}" in
  artaxor)
    CONFIG_DATASET="ArTaxOr"
    OUTPUT_DATASET="ArTaxOr"
    ;;
  clipart1k)
    CONFIG_DATASET="clipart1k"
    OUTPUT_DATASET="Clipart1k"
    ;;
  dior)
    CONFIG_DATASET="DIOR"
    OUTPUT_DATASET="DIOR"
    ;;
  deepfish|fish)
    CONFIG_DATASET="FISH"
    OUTPUT_DATASET="FISH"
    ;;
  neu-det|neudet)
    CONFIG_DATASET="NEU-DET"
    OUTPUT_DATASET="NEU-DET"
    ;;
  uodd)
    CONFIG_DATASET="UODD"
    OUTPUT_DATASET="UODD"
    ;;
  *)
    echo "[ERROR] Unsupported dataset: ${DATASET}" >&2
    echo "        Choose ArTaxOr, Clipart1k, DIOR, FISH/DeepFish, NEU-DET, or UODD." >&2
    exit 2
    ;;
esac

case "${SHOT}" in
  1|5|10) ;;
  *)
    echo "[ERROR] Unsupported shot: ${SHOT}. Choose 1, 5, or 10." >&2
    exit 2
    ;;
esac

if [[ "${GPU_ID}" != "0" ]]; then
  echo "[ERROR] This RTX 5090 repository exposes physical GPU 0 only; use --gpu 0." >&2
  exit 2
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "[ERROR] --port must be an integer from 1 to 65535." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${REPO_ROOT}/configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_${CONFIG_DATASET}_${SHOT}shot.py"
WORK_DIR="${REPO_ROOT}/exp_cdfsod_results/${OUTPUT_DATASET}/${SHOT}shot"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

TRAIN_ARGS=()
if [[ "${RESUME}" == true ]]; then
  TRAIN_ARGS+=(--resume)
fi

echo "Dataset : ${OUTPUT_DATASET}"
echo "Shot    : ${SHOT}"
echo "GPU     : ${GPU_ID}"
echo "Config  : ${CONFIG}"
echo "Output  : ${WORK_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "[OK] Dry run completed."
  exit 0
fi

mkdir -p "${WORK_DIR}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

"${REPO_ROOT}/tools/dist_train.sh" \
  "${CONFIG}" 1 "${PORT}" "${GPU_ID}" \
  --work-dir "${WORK_DIR}" "${TRAIN_ARGS[@]}"

shopt -s nullglob
BEST_CHECKPOINTS=("${WORK_DIR}"/best_coco_bbox_mAP_iter_*.pth)
shopt -u nullglob

if (( ${#BEST_CHECKPOINTS[@]} == 0 )); then
  echo "[ERROR] Training finished without best_coco_bbox_mAP_iter_*.pth in ${WORK_DIR}." >&2
  exit 1
fi

BEST_CHECKPOINT="${BEST_CHECKPOINTS[0]}"
echo "Best checkpoint: ${BEST_CHECKPOINT}"

"${REPO_ROOT}/tools/dist_test.sh" \
  "${CONFIG}" "${BEST_CHECKPOINT}" 1 "${PORT}" "${GPU_ID}" \
  --work-dir "${WORK_DIR}" \
  --out "${WORK_DIR}/results.pkl"

echo "[OK] ${OUTPUT_DATASET} ${SHOT}-shot training and evaluation completed."
