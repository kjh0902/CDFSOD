#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-NEU-DET}"
SHOT="${2:-1}"
GPUS="${3:-1}"
# Example: CDFSOD_SUPPORT_IMAGES_PER_CLASS=2 bash "$0" clipart1k 10 1
MMDET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MMDET_DIR"
export PYTHONPATH="$MMDET_DIR:${PYTHONPATH:-}"

export CDFSOD_DATA_ROOT="${CDFSOD_DATA_ROOT:-/home/aislab5090/CDFSOD/junhyung/datasets}"
export CDFSOD_DATASET="$DATASET"
export CDFSOD_TRAIN_ANN="annotations/${SHOT}_shot.json"
export CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES="${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES:-1}"
export CDFSOD_CAPTION_FILE="${CDFSOD_CAPTION_FILE:-annotations/${SHOT}_shot_captions.json}"
export CDFSOD_SUPPORT_IMAGES_PER_CLASS="${CDFSOD_SUPPORT_IMAGES_PER_CLASS:-}"

if [[ -n "$CDFSOD_SUPPORT_IMAGES_PER_CLASS" &&
      ! "$CDFSOD_SUPPORT_IMAGES_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CDFSOD_SUPPORT_IMAGES_PER_CLASS must be a positive integer." >&2
  exit 2
fi

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py"
TEXT_TAG="class_name_token_prototype"
if [ "$CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES" = "0" ]; then
  TEXT_TAG="class_name"
fi
SUPPORT_IMAGE_TAG=""
if [ -n "$CDFSOD_SUPPORT_IMAGES_PER_CLASS" ]; then
  SUPPORT_IMAGE_TAG="_support${CDFSOD_SUPPORT_IMAGES_PER_CLASS}img_per_class"
fi
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}${SUPPORT_IMAGE_TAG}"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "class-name token prototypes: ${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES}"
echo "caption file: ${CDFSOD_CAPTION_FILE}"
echo "support images per class: ${CDFSOD_SUPPORT_IMAGES_PER_CLASS:-all}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  python tools/train.py "$CONFIG" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py "$CONFIG" "$WORK_DIR/epoch_30.pth" --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh "$CONFIG" "$WORK_DIR/epoch_30.pth" "$GPUS" --work-dir "$WORK_DIR"
fi
