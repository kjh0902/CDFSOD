# CDFSOD Grounding DINO attribute-selected class-name token prototype

이 저장소는 CDFSOD few-shot detection을 위한 MMDetection 기반 Grounding DINO
학습 코드입니다. 클래스별 K-shot support instance의 GT bbox crop들로 5개
attribute-wise visual description을 생성하고, pretrained Grounding DINO의
Language-Guided Query Selection score로 Top-1을 offline 선택합니다. 학습에서는
선택된 prompt의 class-name token feature만 평균하여 text prototype을 만듭니다.
`[CLS]` token과 description token 자체는 prototype으로 사용하지 않습니다.

## 데이터 구조

기본 데이터 루트는 `/home/aislab5090/CDFSOD/junhyung/datasets`입니다.

```text
DATASET_NAME/
  annotations/
    train.json
    test.json
    1_shot.json
    1_shot_caption_candidates.json
    1_shot_captions.json
    1_shot_caption_selection.log
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

## 1. Attribute-wise description 후보 생성

먼저 클래스별 Shape, Texture, Boundary, Internal Structure,
Color/Intensity/Material description을 생성합니다. 각 attribute inference에는 같은
클래스의 전체 K-shot GT bbox crop이 함께 입력됩니다. 원본 이미지와 bbox 좌표는
Qwen에 전달되지 않습니다.

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_caption_candidates.json
```

생성 스크립트는 기본적으로 `Qwen/Qwen3-VL-8B-Instruct`를 사용하며, CUDA를 사용할
수 없으면 CPU로 전환합니다. 다른 Qwen3-VL checkpoint나 장치를 사용하려면 각각
`--model-name`, `--device`로 지정할 수 있습니다. 각 출력은 한 문장이고 class name을
직접 포함하지 않는지 검사한 뒤 `candidates`에 저장됩니다.

## 2. Pretrained Grounding DINO description 선택

Fine-tuning 전에 다음 명령을 한 번 실행합니다. 대상 class는 candidate description으로
문맥화한 class-name token 평균 prototype을 사용하고, 나머지 class는 description 없는
baseline prototype을 사용합니다. Top-900 proposal 중심이 support GT bbox 내부이면서
image-text similarity의 argmax prototype이 대상 class인 query 수를 object별로 세고,
K-shot GT object 평균이 가장 큰 description을 선택합니다.

```bash
cd mmdetection
python tools/select_attribute_descriptions.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --candidate-file annotations/1_shot_caption_candidates.json \
  --output annotations/1_shot_captions.json
```

선택 결과 JSON에는 attribute별 `scores`, `selected_attribute`,
`selected_description`과 기존 학습 호환용 `caption`이 저장됩니다. 기본 log 파일은
`annotations/1_shot_caption_selection.log`입니다. 선택은 학습 loop와 분리되어 있어
epoch마다 다시 실행되지 않습니다.

## 3. 학습과 평가

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
- support candidates: `annotations/{SHOT}_shot_caption_candidates.json`
- selected visual description: `annotations/{SHOT}_shot_captions.json`
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
