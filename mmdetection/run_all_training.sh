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

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py"
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  python tools/train.py "$CONFIG" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py "$CONFIG" "$WORK_DIR/epoch_30.pth" --work-dir "$WORK_DIR"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --amp --work-dir "$WORK_DIR"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh "$CONFIG" "$WORK_DIR/epoch_30.pth" "$GPUS" --work-dir "$WORK_DIR"
fi
