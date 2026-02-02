#!/bin/bash
BASE_CONFIG="configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py"
for DATASET in ArTaxOr clipart1k NEU-DET FISH; do
  for SHOT in 1 5 10; do
    FEWSHOT_CONFIG="configs/grounding_dino/CDFSOD_detection_few-shot_${DATASET}_${SHOT}shot.py"
    python tools/train.py $BASE_CONFIG --cfg-options _base_.0=$FEWSHOT_CONFIG
  done
done