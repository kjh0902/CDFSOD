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
export CDFSOD_USE_ENRICHED_CLASS_TOKENS="${CDFSOD_USE_ENRICHED_CLASS_TOKENS:-${CDFSOD_USE_CLASS_PROTOTYPES:-1}}"
export CDFSOD_CAPTION_FILE="${CDFSOD_CAPTION_FILE:-annotations/${SHOT}_shot_captions.json}"

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py"
TEXT_TAG="enriched_class_tokens"
if [ "$CDFSOD_USE_ENRICHED_CLASS_TOKENS" = "0" ]; then
  TEXT_TAG="class_name"
fi
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "enriched class tokens: ${CDFSOD_USE_ENRICHED_CLASS_TOKENS}"
echo "caption file: ${CDFSOD_CAPTION_FILE}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  python tools/train.py "$CONFIG" --amp --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --amp --work-dir "$WORK_DIR"
fi
