# CDFSOD Grounding DINO BLIP-2 visual prototype

이 저장소는 CDFSOD few-shot detection을 위한 MMDetection 기반 Grounding DINO
학습 코드입니다. 기본 설정은 클래스별 K-shot support instance의 GT bbox crop을
pretrained BLIP-2 Image Encoder와 Q-Former에 입력하여 class당 32개 visual prototype
token을 만듭니다. CDFSOD prototype 생성 경로에서는 BERT를 사용하지 않으며, 일반
Grounding DINO text prompt 경로는 그대로 유지합니다.

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
`mmdetection/configs/_base_/datasets/CDFSOD_detection_few-shot.py`에 정의되어
있습니다.

## support object annotation

별도의 caption 파일은 필요하지 않습니다. 학습에 선택된 표준 COCO annotation
`annotations/{SHOT}_shot.json`의 `images`, `categories`, `annotations`를 직접 읽습니다.
각 annotation의 `image_id`로 train image의 `file_name`을, `category_id`로 class name을
찾고 `bbox`를 support crop으로 사용합니다. 모든 target class에는 최소 한 개의 support
object가 있어야 합니다. Prototype loader는 `support_image_root` 아래의 train image만
읽으며 test image와 test GT는 사용하지 않습니다.

## BLIP-2 visual prototype

기본 checkpoint는 `Salesforce/blip2-itm-vit-g`입니다. 이 checkpoint의 pretrained
ViT-G Image Encoder, pretrained Q-Former, pretrained 32 query tokens만 사용하며 language
model과 BLIP-2 projection은 사용하지 않습니다. 다른 Hugging Face model id 또는 로컬
checkpoint 경로는 `CDFSOD_BLIP2_MODEL`로 지정할 수 있습니다.

GT bbox crop은 BLIP-2 checkpoint의 image processor로 resize/normalize됩니다. Object별
Q-Former 출력은 `[32,768]`이며 같은 class의 K-shot 출력 `[K,32,768]`은 shot dimension만
평균합니다. 32 query token은 평균하지 않습니다. 결과 `[C,32,768]`은 Grounding DINO
checkpoint의 기존 pretrained `text_feat_map`으로 `[C,32,256]`에 투영됩니다. 새 projection
layer는 만들지 않습니다.

학습에서는 BLIP-2 전처리 pixel tensor만 CPU에 보관하고 Image Encoder와 Q-Former 출력을
매 iteration 다시 계산합니다. Image Encoder, Q-Former, query tokens, `text_feat_map` 모두
fine-tuning됩니다. 평가에서는 마지막 checkpoint의 parameter와 target support set으로
prototype을 첫 predict 시 다시 생성해 detach/cache합니다.

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

학습 annotation과 BLIP-2 support annotation은 동일한
`CDFSOD_TRAIN_ANN=annotations/{SHOT}_shot.json`입니다.

기본 실험 결과는
`mmdetection/work_dirs/{DATASET}_{SHOT}shot_blip2_visual_prototype`에 저장됩니다.
스크립트는 30 epoch 학습 후 `epoch_30.pth`를 평가합니다.

BLIP-2 visual prototype을 사용하지 않고 class name만 사용하는 Grounding DINO baseline은
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
- learning rate: `1e-4` (backbone `0.1`, BLIP-2 ViT-G `0.01`, Q-Former와
  query tokens `0.1`, `text_feat_map` `0.1` multiplier)
- weight decay: `1e-4`
- batch size: GPU당 `2`
- epochs: `30`
- LR milestone: epoch `20`, gamma `0.1`
- train annotation: `annotations/{SHOT}_shot.json`
- support object annotation: train과 동일한 `annotations/{SHOT}_shot.json`
- BLIP-2: `Salesforce/blip2-itm-vit-g`, 32 queries, hidden dim `768`
- support crop batch size: `2`, pretrained image processor 사용
- prototype tokens: class당 `32`, `max_text_len=max(256, classes*32)`
- validation/test annotation: `annotations/test.json`

상세 설정은
`mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot.py`와
`GroundingDINO-few-shot-SwinB.py`에서 확인할 수 있습니다.

직접 평가할 때는 학습 때 사용한 환경 변수를 동일하게 지정합니다.

```bash
cd mmdetection
CDFSOD_DATASET=NEU-DET \
CDFSOD_TRAIN_ANN=annotations/1_shot.json \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_blip2_visual_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_blip2_visual_prototype
```
