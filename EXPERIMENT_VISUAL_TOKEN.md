# ACL Support Visual Memory

기준은 `acl-vistoken`의 `72bced2`이다. 브랜치는
`acl-vistoken-support-memory`이며, HED 없는 기존 순차 ACL Decoder, DN,
Query 증강, Detection Loss와 optimizer/scheduler/checkpoint 정책을 유지한다.
기존 Text Attention의 학습 parameter와 checkpoint key를 그대로 사용하며,
새 학습 parameter나 보조 Loss를 추가하지 않는다.

## Support 객체 Memory

18개 finetuning config의 Support dataset은 각 Shot의 학습 annotation·이미지·
클래스 목록을 사용한다. 평가 GT는 Memory 생성에 사용하지 않는다. 기존 COCO
파싱의 invalid/ignore/crowd 처리를 유지하며, 일반 GT instance만 추출한다.
Query sampler와 학습 이미지 필터는 Support에 적용하지 않는다.

Support pipeline은 LoadImage → LoadAnnotations → 고정 Resize(800,1333) →
Pack이다. 동일한 preprocessor의 정규화·padding 후 resized GT xyxy로
정규화된 Swin-B `norm1` 반환값(stride 8, 256 channels, Neck 이전)을
RoIAlign(7×7, aligned=True, sampling_ratio=2, spatial_scale=1/8) → GAP한다.

객체마다 FP32 Token 한 개를 보존한다. 클래스별 평균, 이미지별 평균, projection,
L2 정규화, Shot별 자르기·반복은 없다. `support_memory()`의 반환값은:

| 필드 | 크기 | 의미 |
|---|---|---|
| tokens | V × 256 | 클래스별로 연결한 모든 유효 GT 객체 Token |
| class_ids | V | 각 Token의 0-based dataset label |
| counts | C | 클래스별 실제 객체 수; 합계가 V |

같은 Shot이라도 여러 객체가 있는 이미지에서는 클래스별 Token 수가 Shot과
다를 수 있다. 클래스 메타데이터, COCO category→label 순서와 label 범위를
검증한다. 특정 클래스의 유효 GT가 0개이면 클래스 index와 이름을 포함한 오류로
중단한다. 빈 이미지는 Token을 추가하지 않는다.

매 학습 iteration마다 각 rank가 전체 Support를 batch size 1, worker 0,
고정 순서로 다시 추출한다. 전용 DataLoader Generator는 Query RNG를 소모하지
않는다. `no_grad`와 Backbone 임시 eval을 사용하고 모든 submodule의 원래 mode를
복구한다. Query Backbone은 정상적으로 학습한다.

Validation/Test에서는 현재 가중치로 첫 predict 시 한 번 생성해 캐시한다.
평가 시작, train/eval 전환과 가중치 재로딩 시 캐시를 무효화한다.
Memory와 통계는 state_dict에 포함하지 않는다.
`support_visual_cfg=None`은 기존 ACL 경로를 사용한다.

## Text Attention과 Mask

BERT → text_feat_map으로 생성한 Text 시퀀스를 그대로 유지한다. 이름 Token
Mapping, Text 위치 ID와 Text 길이를 이동시키지 않는다. Visual Memory는
별도 Tensor이므로 `max_text_len`을 소비하지 않는다. Text 자체의 길이 한도는
기존대로 검증한다. 기존 완전한 canonical 전체 클래스 prompt 검증을 유지하며,
Support 활성화 상태의 chunked/부분 클래스/비정형 recognition prompt는 지원하지 않는다.

각 GroundingDINO Encoder Layer에서 기존 순서대로 Image-Text Fusion →
Text Attention → Image Attention을 수행한다. Text Attention만 다음과 같다.

```text
Q = current Text
K = Concat(current Text, fixed Support Memory)
V = Concat(current Text, fixed Support Memory)
output length = original Text length
```

Q와 Text Key에는 기존 Text 위치 인코딩을 사용한다. Visual Key의 위치
인코딩은 0이며 Shot·객체 순서 인코딩을 추가하지 않는다. 기존
MultiheadAttention, FFN, Norm과 잔차는 Text 행에만 적용한다.
Support는 모든 Layer에서 동일한 원본 Token이며 Query로 사용하거나 갱신하지 않는다.

허용 Mask는 B × T × (T+V)이다. 앞의 T × T는 기존 Text Mask를 그대로
복사한다. 각 Visual 열은 해당 클래스 마침표 열의 기존 허용 연결을 상속하고,
padding Query에서는 차단한다. 따라서 클래스 이름과 period는 해당 클래스
Memory를 볼 수 있고, 다른 클래스·CLS/SEP·padding에는 새로운 연결이 없다.
Text padding의 기존 self 연결은 유지해 완전히 차단된 Query 행으로 인한 NaN을
피한다. 유효 Query는 Text padding Key를 보지 않는다. Support 자체에는 padding이 없다.

Attention 호출 시 Batch별 Mask를 head 수만큼 repeat_interleave해
B×heads 순서로 확장한다. 서로 다른 Batch prompt 길이에서도 head가 다른
이미지의 Mask를 사용하지 않는다.

## Fusion, Detection과 checkpoint

Bidirectional Image-Text Fusion에는 기존 Text만 전달한다. 앞 Layer에서
Support로 보강된 Text가 다음 Layer의 정상 Fusion에 참여하는 간접 효과는 유지한다.
Visual Memory를 Image의 양방향 Attention에 직접 전달하지 않는다.

Encoder Query Selection, Decoder Text Cross-Attention과 Contrastive Head는
기존 길이의 Text Feature만 받는다. Visual/Text Max Fusion, Visual Score와
클래스 one-hot으로 바꾸던 전용 경로를 제거했다.

Detection Head는 `449e69a`의 기존 serial ACL 구현과 동일하다. Encoder Top-K는
기존 유효 Text Token Logit의 max를 사용한다. 모든 Decoder Layer, DN,
선택된 Encoder Query의 Focal/L1/GIoU와 Hungarian Matching은 원래
Token positive map과 Text mask를 사용한다. 추론은 기존 이름 Token의 sigmoid
점수 평균으로 클래스 점수를 구한다.

기존 Attention·FFN·Norm parameter 이름과 shape가 동일하므로 기존 checkpoint를
읽을 수 있다. 이전 Visual 삽입/Max Fusion 모델 checkpoint도 parameter 구조는
호환되지만, 추론 동작은 이 브랜치의 Memory 구조로 바뀐다.

## 통계: schema_version 2

기존 `VisualTokenHook`과 `visual_token_stats.jsonl` 파일명을 유지한다.
각 행은 `schema_version=2`, `architecture=support_memory`와
Dataset/Shot/phase/Epoch/Iteration을 포함한다.

| 항목 | 정의 |
|---|---|
| support_token_count | 전체/클래스별 실제 Support 객체 수. 이미지·Batch·rank 수로 곱하지 않음 |
| text_norm | text_feat_map 이후, FE 이전 이름 Token의 평균 L2 Norm |
| support_token_norm | 원본 Support 객체 Token의 평균 L2 Norm |
| enhanced_text_norm | 마지막 Encoder 이후 이름 Text Token의 평균 L2 Norm |

Norm은 `{value,count}`로 기록하며 count는 실제 집계 Token 관측 수다.
Support Norm은 각 유효 Query 이미지에 대해 객체별로 집계한다.
같은 평가 동안 Memory는 고정이므로 결과는 객체별 평균 Norm과 동일하다.
분모가 0이면 value는 null이다.

Visual 승률, Visual Score, Score 차이와 Visual Query Selection 통계는
제거했다. 통계를 위한 추가 Hungarian Matching과 모델 Forward도 없다.
학습은 Epoch 마지막 Batch, Validation/Test는 전체 dataset을 집계한다.
DDP는 Norm의 합계·분모만 합산하며 Support 객체 수는 rank별 전체 Set의
snapshot을 그대로 사용한다. unshuffled DefaultSampler의 평가 padding
중복은 모든 Norm 집계에서 제외한다. 로깅 on/off는 Loss·gradient·RNG에
영향을 주지 않는다.

## 실행과 검증

기존 실행 방법과 18개 dataset/shot 설정을 그대로 사용한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1
python -B -m unittest discover -s tests -v
```

CPU 테스트는 실제 torchvision RoIAlign, repository의 Encoder/Text Layer/Fusion
forward, PyTorch Attention, Detection Head Loss·Hungarian 비용과 scipy
assignment를 사용한다. 이미지의 deformable attention과 BERT/Backbone은
소형 fixture로 대체한다. MMCV 또는 mmcv-lite와 MMEngine이 있으면 실제
MMCV Attention/FFN/Norm으로 256차원 FP32·CPU autocast forward/backward와
checkpoint parameter 호환성도 검사한다.

실제 MMEngine/pycocotools로 COCO 초기화 후 임시 coco 속성이 삭제되는 lifecycle,
비연속 category ID 및 class order를 검사하고 18개 config를 파싱한다.
원래 config·순차 Decoder·학습 Hook이 변경되지 않았는지도 확인한다.

로컬 최종 결과는 **43개 테스트 통과, 실패 0개, 건너뜀 0개**이다.
MMEngine 0.10.7, mmcv-lite 2.2.0, pycocotools 2.0.11을 별도 테스트 전용
폴더에서 사용했으며, 기본 학습 의존성과 환경 설정은 변경하지 않았다.
18개 config 파싱, 실제 Detection Loss를 통한 Memory→Text Attention
backward, parameter key/shape 호환성 및 로깅 on/off 동등성 검사를 포함한다.

전체 CUDA 모델 검사는 MMCV CUDA ops, BERT, pretrained checkpoint 및 해당
Shot의 실제 Support 이미지·annotation이 준비된 환경에서만 실행한다.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/check_visual_tokens.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py
```

위 도구는 synthetic Query 한 Batch의 실제 Detection Loss/backward와 predict를
수행하며 optimizer step이나 checkpoint 저장은 하지 않는다. 전체 학습 실험은
검증 대상에 포함하지 않는다.

검증 환경은 Windows / Python 3.13 / PyTorch 2.7.1+cpu /
torchvision 0.22.1+cpu이다. 실제 CUDA 전체 모델, 실제 Support 이미지 기반
학습·추론, mAP·처리량·NCCL 종료는 미검증이다. 매 iteration 전체 Support를
재계산하는 비용과 T×(T+V) Text Attention 비용은 실제 객체 수에 비례해 증가한다.

기존 train/test의 distributed cleanup 및 warn_only 호환 처리는 유지한다.
