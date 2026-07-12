# CDFSOD GroundingDINO 사용법

이 폴더는 MMDetection 기반 Grounding DINO 코드를 담고 있습니다.
자세한 설치와 전체 실험 흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

support caption 생성:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json
```

class-level text prototype 방식 학습:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
```

class name만 사용하는 baseline:

```bash
CDFSOD_USE_CLASS_PROTOTYPES=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력:

```bash
CDFSOD_DEBUG_TEXT_PROTOTYPE=1 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

best checkpoint test:

```bash
cd mmdetection
python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_class_prototype/best_coco_bbox_mAP_epoch_*.pth
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_CLASS_PROTOTYPES=1`입니다.

학습 중 support caption JSON을 한 번 읽어서 class별 prompt bank를 만들고, 매 iteration마다 현재 BERT로 전체 support prompt를 다시 encoding합니다. 같은 class에 속한 support prompt들의 `[CLS]` feature를 평균해서 class text prototype으로 사용합니다.

baseline을 원하면 `CDFSOD_USE_CLASS_PROTOTYPES=0`으로 실행하면 됩니다.
