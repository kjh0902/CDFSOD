# CDFSOD Grounding DINO baseline

이 저장소는 CDFSOD few-shot detection을 위한 일반 Grounding DINO 학습 코드입니다.
학습 시 매 iteration마다 고정 resize된 전체 support set의 모든 GT object에서 visual
token을 다시 생성하고 detection loss로 textualizer를 포함한 Grounding DINO와 BERT
전체를 학습합니다. Random flip/crop은
detection training image에만 적용됩니다. 평가 시에는 최종 checkpoint로 같은 support
token을 한 번 생성해 캐시하고 모든 test image의 BERT 입력에 공통으로 재사용합니다.

## 데이터 구조

기본 데이터 루트는 `/home/aislab5090/CDFSOD/junhyung/datasets`입니다.

```text
DATASET_NAME/
  annotations/
    train.json
    test.json
    1_shot.json
    5_shot.json
    10_shot.json
  train/
  test/
```

지원 데이터셋은 `NEU-DET`, `clipart1k`, `UODD`이며 class name은
`mmdetection/configs/_base_/datasets/CDFSOD_detection_few-shot.py`에 정의되어 있습니다.

## 학습과 평가

```bash
bash mmdetection/run_all_training.sh DATASET SHOT GPU_COUNT
```

예시:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
bash mmdetection/run_all_training.sh clipart1k 5 4
bash mmdetection/run_all_training.sh UODD 10 4
```

데이터 루트가 다르면 환경 변수로 지정합니다.

```bash
CDFSOD_DATA_ROOT=/other/datasets \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

실험 결과는 `mmdetection/work_dirs/{DATASET}_{SHOT}shot`에 저장됩니다.
스크립트는 30 epoch 학습 후 `epoch_30.pth`를 평가합니다.

## 기본 설정

- 모델: Grounding DINO Swin-B
- pretrained checkpoint: OpenMMLab Grounding DINO Swin-B checkpoint
- optimizer: AdamW
- learning rate: `1e-4` (backbone multiplier `0.1`)
- weight decay: `1e-4`
- batch size: GPU당 `2`
- epochs: `30`
- LR milestone: epoch `20`, gamma `0.1`
- train annotation: `annotations/{SHOT}_shot.json`
- validation/test annotation: `annotations/test.json`

상세 설정은 `mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot.py`와
`GroundingDINO-few-shot-SwinB.py`에서 확인할 수 있습니다.

직접 평가할 때는 다음 명령을 사용합니다.

```bash
cd mmdetection
CDFSOD_DATASET=NEU-DET \
CDFSOD_TRAIN_ANN=annotations/1_shot.json \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot
```
