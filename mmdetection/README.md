# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다.
전체 설치와 실험 흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

support visual description 생성:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
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

Qwen3-VL은 같은 클래스의 K-shot GT bbox crop 전체와 class name을 하나의 conversation으로 입력받아 클래스당 common visual description 하나를 생성합니다. 원본 이미지 전체와 bbox 좌표는 Qwen에 전달하지 않으며, instance별 description을 생성하거나 평균하지 않습니다. 학습 중 이 JSON을 한 번 읽어서 class별 prompt bank를 만들고, 매 iteration마다 현재 BERT로 class-level enriched prompt를 다시 encoding합니다.
BERT에는 `{class_name}: {visual_description}.` 전체 prompt가 들어가지만, 출력에서는 tokenizer `offset_mapping`을 이용해 class name 문자 범위와 겹치는 token feature만 선택합니다.

JSON에 클래스당 prompt가 하나이므로 선택된 class-name token feature로 class당 1개 text prototype을 만듭니다. `[CLS]` token은 사용하지 않습니다.
같은 JSON의 `file_names`와 `bboxes`로 support object를 다시 읽고, 현재 backbone+neck의 첫 feature에서 object별 `7x7` RoIAlign token을 추출합니다. Shot 평균 없이 클래스별로 concatenate한 token을 key/value로, text prototype `T [C,256]`를 query `T.unsqueeze(1) [C,1,256]`로 사용하는 단일 8-head cross-attention이 `V [C,256]`를 만듭니다. 최종 prototype은 정규화 없이 정확히 `T + alpha * V`이며, `alpha`는 0으로 초기화되는 learnable scalar이므로 초기 출력은 기존 `T`와 같습니다.

학습 중 support visual feature와 cross-attention 출력은 매 iteration 다시 계산됩니다. 매 epoch 종료 후 `visual_gate_history.json`에 1-based epoch와 현재 `alpha`가 기록됩니다. 평가 중에는 target support set으로 fused prototype을 최초 한 번만 생성해 detach/cache하고 모든 test image에서 재사용하며, test image는 prototype 생성에 사용되지 않습니다.
baseline을 원하면 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0`으로 실행하면 됩니다.
