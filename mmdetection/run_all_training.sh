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

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB-visual-cross-attention.py"
TEXT_TAG="class_name_token_prototype"
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}"
CHECKPOINT="${WORK_DIR}/epoch_30.pth"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "class-name token prototypes: ${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES}"
echo "caption file: ${CDFSOD_CAPTION_FILE}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  # Keep the full model in FP32 for stable joint fine-tuning.
  python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
    "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh \
    "$CONFIG" "$CHECKPOINT" "$GPUS" \
    --work-dir "$WORK_DIR"
fi
