#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-NEU-DET}"
SHOT="${2:-1}"
GPUS="${3:-1}"
if [ "$#" -gt 3 ]; then
  echo "Usage: $0 [DATASET] [SHOT] [GPU_COUNT]" >&2
  exit 2
fi

MMDET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MMDET_DIR"
export PYTHONPATH="$MMDET_DIR:${PYTHONPATH:-}"

export CDFSOD_DATA_ROOT="${CDFSOD_DATA_ROOT:-/home/aislab5090/CDFSOD/junhyung/datasets}"
export CDFSOD_DATASET="$DATASET"
export CDFSOD_TRAIN_ANN="annotations/${SHOT}_shot.json"
export CDFSOD_USE_BLIP_PROTOTYPES="${CDFSOD_USE_BLIP_PROTOTYPES:-${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES:-1}}"

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py"
TEXT_TAG="blip1_enc_prototype"
if [ "$CDFSOD_USE_BLIP_PROTOTYPES" = "0" ]; then
  TEXT_TAG="class_name"
fi
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "BLIP multimodal prototypes: ${CDFSOD_USE_BLIP_PROTOTYPES}"
echo "support annotation: ${CDFSOD_TRAIN_ANN}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  python tools/train.py "$CONFIG" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py "$CONFIG" "$WORK_DIR/epoch_30.pth" --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh "$CONFIG" "$WORK_DIR/epoch_30.pth" "$GPUS" --work-dir "$WORK_DIR"
fi
