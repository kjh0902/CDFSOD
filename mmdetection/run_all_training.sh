#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-NEU-DET}"
SHOT="${2:-1}"
GPUS="${3:-1}"
MMDET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MMDET_DIR"
export PYTHONPATH="$MMDET_DIR:${PYTHONPATH:-}"

export CDFSOD_DATA_ROOT="${CDFSOD_DATA_ROOT:-/home/aislab5090/CDFSOD/junhyung/datasets}"
export CDFSOD_DATASET="$DATASET"
export CDFSOD_TRAIN_ANN="annotations/${SHOT}_shot.json"
export CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES="${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES:-1}"
export CDFSOD_CAPTION_FILE="${CDFSOD_CAPTION_FILE:-annotations/${SHOT}_shot_captions.json}"

if [ "$CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES" = "0" ]; then
  echo "Two-stage training requires the visual textualizer." >&2
  echo "Set CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=1." >&2
  exit 2
fi

STAGE1_CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB-stage1.py"
STAGE2_CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB-stage2.py"
TEXT_TAG="class_name_token_prototype"
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}"
STAGE1_CHECKPOINT="${WORK_DIR}/epoch_5.pth"
STAGE2_CHECKPOINT="${WORK_DIR}/epoch_30.pth"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "class-name token prototypes: ${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES}"
echo "caption file: ${CDFSOD_CAPTION_FILE}"
echo "stage 1 config: ${STAGE1_CONFIG}"
echo "stage 2 config: ${STAGE2_CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  # FP32 avoids the non-finite gradients observed with AMP in both stages.
  python tools/train.py "$STAGE1_CONFIG" --work-dir "$WORK_DIR"
  # Stage 1 is a trusted local MMEngine checkpoint. PyTorch 2.6 otherwise
  # defaults torch.load() to weights_only=True and rejects its HistoryBuffer.
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/train.py \
    "$STAGE2_CONFIG" \
    --work-dir "$WORK_DIR" \
    --cfg-options "load_from=${STAGE1_CHECKPOINT}"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
    "$STAGE2_CONFIG" "$STAGE2_CHECKPOINT" --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$STAGE1_CONFIG" "$GPUS" \
    --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_train.sh \
    "$STAGE2_CONFIG" "$GPUS" \
    --work-dir "$WORK_DIR" \
    --cfg-options "load_from=${STAGE1_CHECKPOINT}"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh \
    "$STAGE2_CONFIG" "$STAGE2_CHECKPOINT" "$GPUS" \
    --work-dir "$WORK_DIR"
fi
