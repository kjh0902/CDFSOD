# CDFSOD Grounding DINO BLIP-1 multimodal prototype

이 저장소는 CDFSOD few-shot detection을 위한 MMDetection 기반 Grounding DINO
학습 코드입니다. 기본 설정은 class name과 클래스별 K-shot support instance의 GT bbox
crop을 pretrained BLIP-1 multimodal encoder에 독립적인 image-text pair로 입력합니다.
BLIP text output을 선택한 aggregation mode로 결합한 뒤 Grounding DINO text prototype으로
사용합니다. 이 prototype 경로에서는 BERT를 사용하지 않으며 일반 Grounding DINO text
prompt 경로는 그대로 유지합니다.

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

## Support object annotation

별도의 caption 파일은 필요하지 않습니다. 학습에 선택된 표준 COCO annotation
`annotations/{SHOT}_shot.json`의 `images`, `categories`, `annotations`를 직접 읽습니다.
각 annotation의 `image_id`로 train image의 `file_name`을, `category_id`로 class name을
찾고 `bbox`를 support crop으로 사용합니다. 모든 target class에는 최소 한 개의 support
object가 있어야 합니다. Prototype loader는 `support_image_root` 아래의 train image만
읽으며 test image와 test GT는 사용하지 않습니다.

## BLIP-1 multimodal prototype

기본 checkpoint는 `Salesforce/blip-itm-base-coco`입니다. 이 checkpoint의 pretrained
vision encoder와 image cross-attention이 있는 multimodal text encoder만 사용하며 ITM
head와 BLIP projection은 사용하지 않습니다. 다른 Hugging Face model id 또는 로컬
checkpoint 경로는 `CDFSOD_BLIP_MODEL`로 지정할 수 있습니다.

각 support crop은 정규화된 class name과 독립적인 pair로 encoding되어 `[L,768]` text
output을 만듭니다. 같은 클래스의 K-shot 결과는 `[K,L,768]`로 stack된 뒤 다음 mode 중
하나로 aggregation됩니다.

- `class_avg`: class-name token만 선택하고 shot과 token 차원을 모두 평균해 `[1,768]`.
- `class_tokens`: class-name token을 선택하고 shot 차원만 평균해 `[S,768]`.
- `all_tokens`: padding을 제외한 `[CLS]`, `[SEP]` 포함 전체 token을 위치별 shot 평균해
  `[L_valid,768]`.

결과는 Grounding DINO checkpoint의 기존 pretrained `text_feat_map`으로 768→256
projection됩니다. 새 projection layer는 만들지 않습니다. 전체 class prototype token
수는 Grounding DINO `max_text_len=256`을 넘을 수 없습니다.

학습에서는 BLIP 전처리 image/text tensor만 CPU에 보관하고 vision encoder와 multimodal
text encoder 출력을 매 iteration 다시 계산합니다. 두 encoder와 `text_feat_map`을 모두
fine-tuning합니다. 평가에서는 마지막 checkpoint와 target support set으로 prototype을
첫 predict 시 다시 생성해 detach/cache합니다.

## 학습과 평가

```bash
bash mmdetection/run_all_training.sh DATASET SHOT GPU_COUNT \
  --blip-prototype-mode class_avg
```

`--blip-prototype-mode` 허용 값은 `class_avg`, `class_tokens`, `all_tokens`이며 기본값은
`class_avg`입니다.

예시:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1 \
  --blip-prototype-mode class_avg
bash mmdetection/run_all_training.sh clipart1k 5 4 \
  --blip-prototype-mode class_tokens
bash mmdetection/run_all_training.sh UODD 10 4 \
  --blip-prototype-mode all_tokens
```

데이터 루트가 다르면 환경 변수로 지정합니다.

```bash
CDFSOD_DATA_ROOT=/other/datasets \
  bash mmdetection/run_all_training.sh NEU-DET 1 1 \
  --blip-prototype-mode class_avg
```

학습 annotation과 BLIP support annotation은 동일한
`CDFSOD_TRAIN_ANN=annotations/{SHOT}_shot.json`입니다. 기본 실험 결과는
`mmdetection/work_dirs/{DATASET}_{SHOT}shot_blip1_{MODE}_prototype`에 저장됩니다.
스크립트는 30 epoch 학습 후 같은 mode로 `epoch_30.pth`를 평가합니다.

BLIP prototype을 사용하지 않고 class name만 사용하는 Grounding DINO baseline은 다음과
같이 실행합니다. 이전 환경변수 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0`도 fallback으로
지원합니다.

```bash
CDFSOD_USE_BLIP_PROTOTYPES=0 \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력을 켜려면 `CDFSOD_DEBUG_TEXT_TOKENS=1`을 지정합니다.

## 기본 설정

- 모델: Grounding DINO Swin-B
- pretrained checkpoint: OpenMMLab Grounding DINO Swin-B checkpoint
- optimizer: AdamW
- learning rate: `1e-4` (backbone `0.1`, BLIP vision encoder `0.01`, BLIP
  multimodal text encoder `0.1`, `text_feat_map` `0.1` multiplier)
- weight decay: `1e-4`
- batch size: GPU당 `2`
- epochs: `30`
- LR milestone: epoch `20`, gamma `0.1`
- train/support annotation: `annotations/{SHOT}_shot.json`
- BLIP-1: `Salesforce/blip-itm-base-coco`, hidden dim `768`
- support pair batch size: `2`, pretrained processor 사용
- prototype mode: `class_avg`
- maximum total prototype tokens: `256`
- validation/test annotation: `annotations/test.json`

상세 설정은
`mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot.py`와
`GroundingDINO-few-shot-SwinB.py`에서 확인할 수 있습니다.

직접 평가할 때는 학습 때 사용한 환경 변수와 prototype mode를 동일하게 지정합니다.

```bash
cd mmdetection
CDFSOD_DATASET=NEU-DET \
CDFSOD_TRAIN_ANN=annotations/1_shot.json \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_blip1_class_avg_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_blip1_class_avg_prototype \
  --blip-prototype-mode class_avg
```
