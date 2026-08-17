# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다.
전체 설치와 실험 흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

attribute-wise support visual description 후보 생성:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_caption_candidates.json
```

pretrained Grounding DINO 기반 Top-1 선택:

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

class-name token prototype 방식 학습:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
```

class name만 사용하는 baseline:

```bash
CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력:

```bash
CDFSOD_DEBUG_TEXT_TOKENS=1 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

30 epoch checkpoint 직접 평가:

```bash
cd mmdetection
python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_class_name_token_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_class_name_token_prototype
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=1`입니다.

Qwen3-VL은 같은 클래스의 전체 K-shot GT bbox crop을 각 attribute prompt에 함께 입력하여 Shape, Texture, Boundary, Internal Structure, Color/Intensity/Material 후보를 한 문장씩 생성합니다. 원본 이미지 전체와 bbox 좌표는 Qwen에 전달하지 않습니다.

별도 selector는 pretrained Grounding DINO로 각 후보를 한 번 평가합니다. 평가 대상 class는 candidate로 문맥화한 class-name token 평균 prototype을, 나머지 class는 description 없는 baseline prototype을 사용합니다. Top-900 proposal 중심이 GT bbox 내부이고 similarity argmax가 대상 class prototype인 query 수를 support object별로 계산한 뒤 평균하여 Top-1을 고정합니다. 결과와 score는 selected JSON 및 `*_caption_selection.log`에 저장됩니다.

학습 중 selected JSON을 한 번 읽어서 class별 prompt bank를 만들고, 매 iteration마다 현재 BERT로 class-level enriched prompt를 다시 encoding합니다. BERT에는 `{class_name}: {selected_description}.` 전체 prompt가 들어가지만, 출력에서는 tokenizer `offset_mapping`을 이용해 class name 문자 범위와 겹치는 token feature만 선택합니다.

JSON에 클래스당 prompt가 하나이므로 선택된 class-name token feature로 class당 1개 text prototype을 만듭니다. `[CLS]` token은 사용하지 않습니다.
baseline을 원하면 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0`으로 실행하면 됩니다.
