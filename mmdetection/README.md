# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다.
전체 설치와 실험 흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

support caption 생성:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json
```

enriched class-name token 방식 학습:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
```

class name만 사용하는 baseline:

```bash
CDFSOD_USE_ENRICHED_CLASS_TOKENS=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력:

```bash
CDFSOD_DEBUG_TEXT_TOKENS=1 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

best checkpoint test:

```bash
cd mmdetection
python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_enriched_class_tokens/best_coco_bbox_mAP_epoch_*.pth
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_ENRICHED_CLASS_TOKENS=1`입니다.

학습 중 support caption JSON을 한 번 읽어서 class별 prompt bank를 만들고, 매 iteration마다 현재 BERT로 전체 support enriched prompt를 다시 encoding합니다.
BERT에는 `{class_name}, {instance_caption}, {domain_attribute}.` 전체 prompt가 들어가지만, 출력에서는 tokenizer `offset_mapping`을 이용해 class name 문자 범위와 겹치는 token feature만 선택합니다.

선택된 class-name token feature는 평균하지 않고, `[CLS]`도 사용하지 않으며, Grounding DINO의 기존 구조처럼 token-level text feature로 유지됩니다.
baseline을 원하면 `CDFSOD_USE_ENRICHED_CLASS_TOKENS=0`으로 실행하면 됩니다.
