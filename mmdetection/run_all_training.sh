#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-NEU-DET}"
SHOT="${2:-1}"
GPUS="${3:-1}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi

BLIP_PROTOTYPE_MODE="class_avg"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --blip-prototype-mode)
      if [ "$#" -lt 2 ]; then
        echo "--blip-prototype-mode requires a value" >&2
        exit 2
      fi
      BLIP_PROTOTYPE_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "$BLIP_PROTOTYPE_MODE" in
  class_avg|class_tokens|all_tokens) ;;
  *)
    echo "Invalid BLIP prototype mode: $BLIP_PROTOTYPE_MODE" >&2
    exit 2
    ;;
esac

MMDET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MMDET_DIR"
export PYTHONPATH="$MMDET_DIR:${PYTHONPATH:-}"

export CDFSOD_DATA_ROOT="${CDFSOD_DATA_ROOT:-/home/aislab5090/CDFSOD/junhyung/datasets}"
export CDFSOD_DATASET="$DATASET"
export CDFSOD_TRAIN_ANN="annotations/${SHOT}_shot.json"
export CDFSOD_USE_BLIP_PROTOTYPES="${CDFSOD_USE_BLIP_PROTOTYPES:-${CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES:-1}}"

CONFIG="configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py"
TEXT_TAG="blip1_${BLIP_PROTOTYPE_MODE}_prototype"
if [ "$CDFSOD_USE_BLIP_PROTOTYPES" = "0" ]; then
  TEXT_TAG="class_name"
fi
WORK_DIR="work_dirs/${DATASET}_${SHOT}shot_${TEXT_TAG}"

echo "dataset: ${DATASET}"
echo "shot: ${SHOT}"
echo "data root: ${CDFSOD_DATA_ROOT}/${DATASET}"
echo "BLIP multimodal prototypes: ${CDFSOD_USE_BLIP_PROTOTYPES}"
echo "BLIP prototype mode: ${BLIP_PROTOTYPE_MODE}"
echo "support annotation: ${CDFSOD_TRAIN_ANN}"
echo "config: ${CONFIG}"
echo "work dir: ${WORK_DIR}"

if [ "$GPUS" = "1" ]; then
  python tools/train.py "$CONFIG" --amp --work-dir "$WORK_DIR" \
    --blip-prototype-mode "$BLIP_PROTOTYPE_MODE"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py "$CONFIG" "$WORK_DIR/epoch_30.pth" --work-dir "$WORK_DIR" \
    --blip-prototype-mode "$BLIP_PROTOTYPE_MODE"
else
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --amp --work-dir "$WORK_DIR" \
    --blip-prototype-mode "$BLIP_PROTOTYPE_MODE"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash tools/dist_test.sh "$CONFIG" "$WORK_DIR/epoch_30.pth" "$GPUS" --work-dir "$WORK_DIR" \
    --blip-prototype-mode "$BLIP_PROTOTYPE_MODE"
fi
