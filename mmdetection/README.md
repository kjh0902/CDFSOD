# CDFSOD GroundingDINO 사용법

이 폴더에는 MMDetection 기반 Grounding DINO 코드가 들어 있습니다. 전체 설치와 실험
흐름은 저장소 루트의 `README.md`를 보면 됩니다.

## 핵심 실행 명령어

Straight-Through BLIP caption prototype 방식 학습:

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
  work_dirs/NEU-DET_1shot_blip1_st_caption_prototype/epoch_30.pth \
  --work-dir work_dirs/NEU-DET_1shot_blip1_st_caption_prototype
```

## 현재 기본 동작

기본값은 `CDFSOD_USE_BLIP_PROTOTYPES=1`입니다.

별도 caption JSON 없이 `annotations/{SHOT}_shot.json`의 표준 COCO `images`,
`categories`, `annotations`를 직접 연결해 K-shot train support object를 crop합니다.
각 crop은 `Salesforce/blip-image-captioning-base`에 독립적으로 입력됩니다. Caption
decoder는 `[DEC]`에서 시작하고 `[DEC]`, `[ENC]` logits를 차단한 noise-free
Straight-Through Argmax로 최대 10 token의 greedy caption을 생성합니다.

BLIP과 Grounding DINO BERT가 공유하는 `bert-base-uncased` vocabulary를 이용해 생성
분포를 BERT `inputs_embeds`로 직접 전달하므로 문자열 decode/re-tokenize로 gradient가
끊기지 않습니다. BERT 입력은 `[CLS] class name : caption [SEP]`이며, BERT 출력 중
class-name subword state만 선택합니다. 같은 class의 support와 subword를 평균한 뒤
기존 `text_feat_map`으로 768→256 projection합니다. 클래스당 하나의 prototype과
positive index를 사용합니다. 학습 중 support mini-batch 전체에 non-reentrant gradient
checkpoint를 적용하며 BLIP captioner와 BERT까지 detection loss의 gradient가 전달됩니다.
평가 중에는 최초 한 번 계산해 detach/cache합니다. Test image와 test GT는 prototype
생성에 사용되지 않습니다.
