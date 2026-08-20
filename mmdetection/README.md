# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다.
전체 설치와 실험 흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

support object metadata 생성:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json
```

BLIP-2 visual prototype 방식 학습:

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
  work_dirs/NEU-DET_1shot_blip2_visual_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_blip2_visual_prototype
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=1`입니다.

Caption JSON의 `file_names`와 `bboxes`로 K-shot train support object를 crop하고,
`Salesforce/blip2-itm-vit-g`의 pretrained image processor, ViT-G Image Encoder,
Q-Former를 적용합니다. Caption text와 BERT는 이 prototype 경로에서 사용하지 않습니다.

각 object의 32개 Q-Former query token을 같은 class의 shot dimension에서만 평균한 뒤,
Grounding DINO checkpoint의 기존 `text_feat_map`으로 768→256 projection합니다. Class당
32개 token 전체가 positive prototype으로 유지됩니다. 학습 중 Image Encoder와 Q-Former
출력은 매 iteration 다시 계산되고, 평가 중에는 마지막 checkpoint로 최초 한 번 계산해
detach/cache합니다. Test image와 test GT는 prototype 생성에 사용되지 않습니다.
baseline을 원하면 `CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=0`으로 실행하면 됩니다.
