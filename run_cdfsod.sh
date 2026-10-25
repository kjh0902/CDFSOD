#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one ACL diagnostic baseline (train with validation each epoch; no checkpoints).

Usage:
  bash run_cdfsod.sh --dataset DATASET --shot {1|5|10} [options]

Required:
  --dataset NAME   ArTaxOr, Clipart1k, DIOR, FISH (or DeepFish), NEU-DET, UODD
  --shot N         1, 5, or 10

Options:
  --gpu ID         Physical GPU 0 or 1; one GPU per run (default: 0)
  --work-dir DIR   Log/diagnostic directory (default: a unique run directory)
  --port PORT      torch.distributed rendezvous port (default: 29500)
  --bert-etf-loss-weight W   Override model.bert_etf_loss_weight (default: 0.0)
  --bert-orth-loss-weight W  Override model.bert_orth_loss_weight (default: 0.0)
  --fe-etf-loss-weight W     Override model.fe_etf_loss_weight (default: 0.0)
  --fe-orth-loss-weight W    Override model.fe_orth_loss_weight (default: 0.0)
                           Diagnostic baseline accepts only zero for every weight.
  --etf-loss-weight W       Legacy alias for --bert-etf-loss-weight
  --resume         Unsupported: this baseline never saves checkpoints
  --dry-run        Validate arguments and print resolved paths without running
  -h, --help       Show this help
EOF
}

DATASET=""
SHOT=""
GPU_ID="0"
PORT="29500"
WORK_DIR=""
BERT_ETF_LOSS_WEIGHT=""
BERT_ORTH_LOSS_WEIGHT=""
FE_ETF_LOSS_WEIGHT=""
FE_ORTH_LOSS_WEIGHT=""
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
    --work-dir)
      [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] --work-dir requires a value." >&2; exit 2; }
      WORK_DIR="$2"
      shift 2
      ;;
    --bert-etf-loss-weight|--etf-loss-weight|--raw-mean-etf-loss-weight)
      [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      BERT_ETF_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --bert-orth-loss-weight)
      [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      BERT_ORTH_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --fe-etf-loss-weight)
      [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      FE_ETF_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --fe-orth-loss-weight)
      [[ $# -ge 2 && -n "$2" ]] || { echo "[ERROR] $1 requires a value." >&2; exit 2; }
      FE_ORTH_LOSS_WEIGHT="$2"
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

if [[ "${GPU_ID}" != "0" && "${GPU_ID}" != "1" ]]; then
  echo "[ERROR] --gpu must select physical GPU 0 or 1 (one GPU per run)." >&2
  exit 2
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "[ERROR] --port must be an integer from 1 to 65535." >&2
  exit 2
fi

if [[ "${RESUME}" == true ]]; then
  echo "[ERROR] --resume is unavailable: diagnostic baseline never saves checkpoints." >&2
  exit 2
fi

CFG_OPTIONS=()
add_loss_weight() {
  local name="$1" weight="$2"
  if [[ -z "${weight}" ]]; then
    echo "${name}: from config"
    return
  fi
  if [[ ! "${weight}" =~ ^\+?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
     ! awk -v w="${weight}" 'BEGIN { n=w+0; exit !(n==0) }'; then
    echo "[ERROR] --${name//_/-} must be zero for the diagnostic baseline." >&2
    exit 2
  fi
  CFG_OPTIONS+=("model.${name}=${weight}")
  echo "${name}: ${weight} (CLI override)"
}
add_loss_weight bert_etf_loss_weight "${BERT_ETF_LOSS_WEIGHT}"
add_loss_weight bert_orth_loss_weight "${BERT_ORTH_LOSS_WEIGHT}"
add_loss_weight fe_etf_loss_weight "${FE_ETF_LOSS_WEIGHT}"
add_loss_weight fe_orth_loss_weight "${FE_ORTH_LOSS_WEIGHT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${REPO_ROOT}/configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_${CONFIG_DATASET}_${SHOT}shot.py"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/exp_cdfsod_results/${OUTPUT_DATASET}/${SHOT}shot/$(date +%Y%m%d_%H%M%S)_$$}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

TRAIN_ARGS=()
if (( ${#CFG_OPTIONS[@]} > 0 )); then
  TRAIN_ARGS+=(--cfg-options "${CFG_OPTIONS[@]}")
fi

echo "Dataset : ${OUTPUT_DATASET}"
echo "Shot    : ${SHOT}"
echo "GPU     : physical ${GPU_ID} -> cuda:0 (single process)"
echo "Config  : ${CONFIG}"
echo "Output  : ${WORK_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "[OK] Dry run completed."
  exit 0
fi

mkdir -p "${WORK_DIR}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

NNODES=1 NODE_RANK=0 bash "${REPO_ROOT}/tools/dist_train.sh" \
  "${CONFIG}" 1 "${PORT}" "${GPU_ID}" \
  --work-dir "${WORK_DIR}" "${TRAIN_ARGS[@]}"

echo "[OK] ${OUTPUT_DATASET} ${SHOT}-shot training and epoch validation completed."
echo "Diagnostics: ${WORK_DIR}/diagnostics"
