# CDFSOD Grounding DINO class-name token prototype

이 저장소는 CDFSOD few-shot detection을 위한 MMDetection 기반 Grounding DINO
학습 코드입니다. 기본 설정은 클래스별 K-shot support instance의 GT bbox crop들로
생성한 common visual description을 BERT prompt로 사용하되, class name에 해당하는
token feature만 선택하여 text prototype을 만듭니다. `[CLS]` token은 사용하지
않습니다.

## 데이터 구조

기본 데이터 루트는 `/home/aislab5090/CDFSOD/junhyung/datasets`입니다.

```text
DATASET_NAME/
  annotations/
    train.json
    test.json
    1_shot.json
    1_shot_captions.json
    5_shot.json
    5_shot_captions.json
    10_shot.json
    10_shot_captions.json
  train/
  test/
```

지원 데이터셋은 `NEU-DET`, `clipart1k`, `UODD`이며 class name은
`mmdetection/configs/_base_/datasets/CDFSOD_detection_few-shot.py`에 정의되어
있습니다.

## support visual description 생성

기본 prototype 방식으로 학습하려면 먼저 클래스별 common visual description JSON을
생성합니다. 스크립트는 같은 클래스의 K개 GT bbox crop과 class name을 하나의
Qwen3-VL conversation에 입력하여 클래스당 description 하나를 생성합니다. bbox
좌표와 원본 이미지 전체는 Qwen에 전달하지 않습니다. 다음 예시는 NEU-DET 1-shot
description을 생성합니다.

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json
```

생성 스크립트는 기본적으로 `Qwen/Qwen3-VL-8B-Instruct`를 사용하며, CUDA를 사용할
수 없으면 CPU로 전환합니다. 다른 Qwen3-VL checkpoint나 장치를 사용하려면 각각
`--model-name`, `--device`로 지정할 수 있습니다. JSON 호환성을 위해 생성된 common
visual description은 기존 `captions` 배열의 `caption` 필드에 클래스당 하나씩
저장됩니다.

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

기본 visual description 파일은 `annotations/{SHOT}_shot_captions.json`입니다. 다른
파일을 사용하려면 다음과 같이 지정합니다.

```bash
CDFSOD_CAPTION_FILE=annotations/custom_captions.json \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

기본 실험 결과는
`mmdetection/work_dirs/{DATASET}_{SHOT}shot_class_name_token_prototype`에 저장됩니다.
스크립트는 30 epoch 학습 후 `epoch_30.pth`를 평가합니다.

visual-description prototype을 사용하지 않고 class name만 사용하는 Grounding DINO baseline은
다음과 같이 실행합니다. 결과 경로에는 `_class_name` suffix가 붙습니다.

```bash
CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0 \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력을 켜려면 `CDFSOD_DEBUG_TEXT_TOKENS=1`을 지정합니다.

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
- support visual description: `annotations/{SHOT}_shot_captions.json`
- validation/test annotation: `annotations/test.json`

상세 설정은
`mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot.py`와
`GroundingDINO-few-shot-SwinB.py`에서 확인할 수 있습니다.

직접 평가할 때는 학습 때 사용한 환경 변수를 동일하게 지정합니다.

```bash
cd mmdetection
CDFSOD_DATASET=NEU-DET \
CDFSOD_TRAIN_ANN=annotations/1_shot.json \
CDFSOD_CAPTION_FILE=annotations/1_shot_captions.json \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_class_name_token_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_class_name_token_prototype
```
