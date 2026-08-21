# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다. 전체 설치와 실험
흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

BLIP-1 multimodal prototype 방식 학습:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
```

class name만 사용하는 baseline:

```bash
CDFSOD_USE_BLIP_PROTOTYPES=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

debug 출력:

```bash
CDFSOD_DEBUG_TEXT_TOKENS=1 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

30 epoch checkpoint 직접 평가:

```bash
cd mmdetection
CDFSOD_DATASET=NEU-DET \
CDFSOD_TRAIN_ANN=annotations/1_shot.json \
python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_blip1_enc_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_blip1_enc_prototype
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_BLIP_PROTOTYPES=1`입니다.

별도 caption JSON 없이 `annotations/{SHOT}_shot.json`의 표준 COCO `images`,
`categories`, `annotations`를 직접 연결해 K-shot train support object를 crop하고,
`Salesforce/blip-itm-base-coco`의 pretrained processor, vision encoder, image
cross-attention이 있는 multimodal text encoder를 적용합니다. BERT는 이 prototype
경로에서 사용하지 않습니다.

각 support crop은 class name과 독립적으로 encoding됩니다. Text sequence 첫 token을
원본 BLIP의 `[ENC]` token으로 바꾸고 image cross-attention을 거친 최종 `[ENC]` hidden
state 하나만 사용합니다. 같은 클래스의 K-shot `[ENC]` feature를 평균한 뒤 기존
`text_feat_map`으로 768→256 projection합니다. 클래스당 하나의 prototype이 Grounding
DINO transformer에 입력되고 training/inference 모두 해당 단일 index를 positive로
사용합니다. 학습 중 BLIP 출력은 매 iteration 다시 계산되고, 평가 중에는 마지막
checkpoint로 최초 한 번 계산해 detach/cache합니다. Test image와 test GT는 prototype
생성에 사용되지 않습니다.
