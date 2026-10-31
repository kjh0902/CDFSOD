# ACL Support Local Descriptor Cross-Attention

이번 수정은 `codex/acl-support-local-cross-attn`의 `944d0d1`을 기준으로 같은 브랜치에 적용한다.
변경은 Support LN과 residual Gamma 초기값 1e-2다. Neck 추출, FE 가중치 이식, Text LN,
내부 1,024차원 확장은 적용하지 않는다. 기준의 HED 없는 순차 ACL
Decoder, DN, Query 증강, Detection Loss, Optimizer/Scheduler, 18개 Dataset/Shot
설정 및 분산 프로세스 종료 처리를 유지한다.

## ACL 2-Stage 학습 정책

활성 Hook `BBoxHeadFirstHook6`에 ACL 원본의 LR=0 Freeze를 복원한다. 기존
Optimizer/ReduceOnPlateau 설정과 Stage 전환 조건은 유지한다. `requires_grad`나
모듈의 train/eval Mode를 바꾸지 않아 고정된 FE를 통한 Support Gradient가 흐른다.

| Parameter Group | Stage 1 | Stage 2 전환 시 |
|---|---|---|
| 기존 FE 및 ACL 원본 Other | LR=0, Weight 갱신 없음 | 당시 Head LR로 Unfreeze |
| encoder.support_layers (Cross-Attention, Support LN 및 Gamma) | 설정된 LR로 학습 | 당시 Head LR로 계속 학습 |
| Backbone / BERT | 기존 설정·Scheduler LR로 학습 | Scheduler LR 유지 |
| Head 관련 (Decoder, Detection Head, Neck, text_feat_map, DN) | 기존 설정·Scheduler LR로 학습 | Scheduler LR 유지 |

MMEngine의 기존 paramwise_cfg가 생성하는 파라미터별 Optimizer Group을 사용하고,
Support를 FE/Other와 분리하여 `acl_group=support`로 식별한다. Support LN도 같은 그룹에서
Stage 1부터 학습하며, 별도의 LR/weight decay 예외는 없다. LN parameter가 추가되므로
LN 없는 과거 Support/Optimizer Checkpoint의 자동 이식이나 Resume은 지원하지 않는다. 서로 다른 정책의 파라미터를
하나의 그룹으로 섞은 사용자 Optimizer는 명시적인 오류로 중단한다. ACL 원본의
Other에는 FE 외에 query_embedding, memory_trans_fc/norm, level_embed 등도 포함된다.

전환 조건은 첫 BERT Group의 현재 LR이 최초 LR의 1/2 이하인 경우다. 매 Epoch 시작에
확인하며 한 번만 전환한다. Plateau patience는 기존 config의 Stage 1=3, Stage 2=8을
따른다. 기본 LR 1e-4 설정에서는 Stage 1 FE=0, Support/Head=1e-4,
Backbone/BERT=2e-5이며, 첫 0.5 감소 후 Stage 2 FE/Support/Head=5e-5,
Backbone/BERT=1e-5가 된다. 이후에는 기존 Scheduler가 모든 그룹을 조정한다.

Resume에서는 Optimizer에 저장된 initial_lr와 현재 BERT LR로 Stage를 복원하고,
Backbone/BERT/Head의 복원된 LR을 초기화하지 않는다. Stage 2를 재동결하지 않으며
Support도 계속 학습한다. Epoch 끝 LR 로그는 FE/Other와 Support를 별도로 표시한다.
사용하지 않는 대체 Hook `StageWiseFreezeHook`의 동작은 유지한다.

## Support Local Memory와 재추출 시점

Support는 해당 Shot 학습 annotation과 이미지에서만 추출한다. 기존 COCO
invalid/ignore/crowd 처리를 유지하고 평가 GT는 사용하지 않는다.
LoadImage → LoadAnnotations → 고정 Resize(800,1333) → Pack 및 동일한
정규화·padding 후, Neck 이전 Swin-B `norm1`(stride 8, 256 channels)에
resized GT xyxy를 적용한다. RoIAlign은 7×7, aligned=True, sampling_ratio=2,
spatial_scale=1/8이다.

GAP 대신 `(N,256,7,7) → (N,49,256)`으로 공간축을 펼친다. 객체별 row/column
순서로 모든 descriptor를 보존한다. Global Descriptor, Projection, L2 정규화,
Selection, Aggregation, Shot별 자르기·반복은 없다.

| Memory 필드 | 크기 | 의미 |
|---|---|---|
| tokens | S × 256 | 클래스별로 연결한 FP32 Local Descriptor |
| class_ids | S | 0-based 클래스 ID, 클래스 순으로 정렬 |
| counts | C | 클래스별 descriptor 수; GT 객체 수 × 49 |

객체 수가 이미지 수/Shot과 다를 수 있다. 각 클래스에 유효 GT가 하나 이상 있어야
하며, 잘못된 Label·Box·클래스 순서는 오류로 처리한다.

Training에서는 `support_memory()`가 매 Iteration/모델 Loss Forward마다 현재
Backbone Weight로 **전체 Support**를 다시 추출한다. 학습 중 descriptor를 캐시하지
않는다. 추출한 Memory는 그 Forward의 모든 FE Layer에서 동일하게 사용하며,
Stop-gradient를 유지한다. `no_grad`, Backbone 임시 eval, 모든 submodule의 원래
mode 복원 및 독립 DataLoader RNG는 그대로다. Query Backbone은 정상적으로 학습한다.

Validation은 `VisualTokenHook.before_val_epoch`, Test는 `before_test_epoch`에서
현재 로드된 Weight로 한 번 재추출하고 해당 평가의 모든 Batch가 재사용한다.
각 rank가 전체 Support를 독립적으로 추출한다. 로깅을 꺼도 재추출은 수행한다.
MMEngine은 이 Hook 이후에 model.eval()을 호출하므로 eval 전환은 방금 생성한
평가 Memory를 유지한다. 다시 train()으로 전환하거나 Weight를 로드하면 평가
Memory를 무효화한다. 다음 Validation Epoch에서는 반드시 새로 추출한다.

Best Checkpoint를 로드한 뒤 Test 평가 시작 Hook이 해당 Backbone으로 재추출한다.
Resume도 재개된 Weight를 로드한 뒤 첫 Training Forward부터 매번 재추출한다.
Memory는 state_dict의 parameter/buffer나 Checkpoint metadata에 저장하지 않으며,
디스크 파일의 저장·로딩·hash 검증·provenance 관리는 사용하지 않는다. 이전 실험의
Support 디스크 캐시가 남아 있어도 읽지 않는다.

Runner 없이 직접 predict하는 경우 eval() 후 `refresh_support_memory()`로 평가
시작을 명시할 수 있다. 준비된 평가 Memory가 없으면 최초 predict에서 한 번
추출한다. `clear_support_cache()`로 메모리 내 평가 결과를 해제할 수 있다.

## Feature Enhancer

각 Layer는 다음 순서로 실행된다.

1. 기존 Bidirectional Image–Text Fusion
2. Support-to-Text Cross-Attention 및 `T + gamma * delta`
3. 원래 Text-only Self-Attention + FFN
4. 기존 Image Deformable Self-Attention + FFN

Q는 Fusion 직후의 **원본 Text**, K/V는 해당 클래스의 Local Descriptor를 각 Layer의
독립적인 Support LN에 통과시킨 값이다. LN은 descriptor별 256개 채널에 적용하며
`eps=1e-5, elementwise_affine=True`, weight=1, bias=0으로 기본 초기화한다.
`local_raw=support_slice.detach()` 이후에 LN을 적용하므로 추출 경로는 Stop-gradient인 반면
LN은 학습된다. Memory에는 LN 이전 descriptor를 보관하며 LN 결과를 평가 cache에 저장하지 않는다.
클래스 이름 Token과 그 period만 같은 클래스 Memory를 볼 수 있고 CLS/SEP/Padding은
Support residual을 받지 않는다. 여러 Batch의 같은 클래스 Query를 묶어 Attention을
계산한다. Query끼리 Attention하지 않으므로 Dense Class-wise Mask와 수학적으로
동일하다. 모든 클래스 descriptor를 사용하면서 전체 B×Head×T×S Mask를 피한다.

Attention은 기존 Text Attention과 동일한 차원/Head 수, Dropout=0으로 구성한다.
Q/K/V/출력 projection은 기존 랜덤 초기화를 유지하고, Channel-wise Gamma는 `1e-2`로 초기화한다.
LN의 기본 초기화는 난수를 소비하지 않는다. 고정 seed에서 기존 projection 초기값과
DINO Xavier 이후의 초기값을 보존한다. DINO는 2차원 이상 Weight만 변경하여
Gamma 및 LN 기본값을 유지한다. Text/Output LN, 추가 gate/temperature, 위치 인코딩,
Target–Support Attention은 없다. Residual은 LN을 거치지 않은 원본 Text에 더한다. 일반 Forward는
`need_weights=False`로 SDPA 경로를 사용하고 Support는 detach한다.

Text Self-Attention의 K/V 결합과 공통 DETR Layer의 explicit K/V 확장은 제거했다.
Support 활성 경로의 Text Mask는 Batch×Head 순서로 repeat_interleave한다. Support
비활성 경로는 원래 repeat 동작까지 그대로 보존한다. 클래스가 다른 Support 변경의
직접 영향은 Cross-Attention에서 차단한다. 이후 정상 Image–Text Fusion을 통한
간접 영향은 모델의 원래 동작이다.

새 parameter는 `encoder.support_layers.*` 아래에만 추가한다. 기존 Weight의 Key와
Shape를 유지하며, 기존 Pretrained 로딩에서 새 parameter만 누락될 수 있다.
`support_visual_cfg=None`이면 새 모듈을 만들지 않고 기존 Forward/Backward를 따른다.

최초 학습은 기존 Grounding DINO pretrained checkpoint에서 시작하며 Support는
랜덤 projection, 기본 LN, Gamma=1e-2를 유지한다. FE 가중치를 복사하지 않는다.
새 Support checkpoint는 LN과 학습된 Gamma를 그대로 복원하며 Resume/Test에서 초기화하지 않는다.
Support key가 일부만 있거나 LN이 없는 과거 checkpoint이면 strict=False에서도 명시적 오류로
중단한다. 기존 Optimizer state를 새 parameter 구성으로 자동 변환하지 않는다.

## 경량 통계: schema_version 3

architecture는 `support_local_cross_attention_support_ln`으로 구분한다. 기존 통계 구조와
schema_version=3, 수집 주기 및 DDP 집계 정책은 유지한다.
`visual_token_stats.jsonl`과 기존 text_norm, support_token_norm,
enhanced_text_norm을 유지한다. support_token_norm은 **LN 이전 Memory**의 norm이며,
Entropy 계산에는 실제 Attention과 동일한 **LN 이후 K 입력**을 전달한다.
Residual 비율의 분모는 원본 Text다. support_token_count는 Local Descriptor 수다.
기존 Norm은 학습 Epoch 마지막 Batch 및 평가 전체 dataset을 집계한다. 같은 평가 동안의 Support
Norm은 한 번 계산한 O(classes) 합계만 재사용한다.

새 `support_cross_attention[layer_id]` 항목은 다음과 같다. Layer ID는 0부터 시작한다.

| 통계 | 정의 |
|---|---|
| normalized_attention_entropy | Head별 `-sum(p log p)/log(S_class)`의 평균. S_class=1이면 0 |
| residual_contribution | 집계된 유효 Token 전체의 `sqrt(sum((gamma*delta)^2))/max(sqrt(sum(T^2)),1e-12)` |
| gamma | Channel별 Gamma의 mean_absolute, max_absolute, 모집단 std |

Entropy/Contribution은 value와 count를 기록한다. 관측이 없으면 value는 null이다.
Entropy count는 Head×샘플 Token 수이고 Contribution count는 유효 Token 수다.

새 통계는 학습 Epoch 마지막 Batch, 평가 Batch index 0/50/100/...에서 수집한다.
Entropy는 각 Layer·클래스·선택된 Batch에서 유효 Query를 Batch/Token 순으로 최대
8개만 택하고, 해당 클래스의 **모든** descriptor를 본다. FP32 no_grad에서 Key를
1,024개씩 나눠 online log-sum-exp와 weighted moment로 정확한 Entropy를 구한다.
Contribution은 선택된 Batch의 모든 유효 클래스 Token을 집계한다. Gamma는
Forward 시점의 Layer별 snapshot이다.

통계는 Class-wise Mask로 유효한 Query와 Key만 사용하고, unshuffled DefaultSampler의
DDP Padding 이미지 중복을 제외한다. DDP는 합계·분모만 합산하고 Gamma 및 전체
Support 수는 rank 0 snapshot을 쓴다. Attention/Feature Tensor를 파일로 저장하지
않으며, 영구 통계 상태는 O(classes+layers)이다. 기존처럼 Epoch/phase당 JSON 한
줄만 기록한다. 로깅 on/off의 Loss·Gradient·Optimizer·RNG 동등성을 검사한다.

## 실행 및 검증

기존 실행 명령을 사용한다. 별도 Support 디스크 캐시 없이 실행한다. 결과는
`exp_cdfsod_results_support_ln_gamma1e2/<dataset>/<shot>shot/`로 분리한다.
`--resume`은 이 LN 구조에서 생성한 checkpoint에만 사용한다.

```bash
bash run_cdfsod.sh --dataset UODD --shot 10
bash run_cdfsod.sh --dataset NEU-DET --shot 10
python analyze_results_cdfsod.py exp_cdfsod_results_support_ln_gamma1e2
python -B -m unittest discover -s tests -v
```

CPU 검증은 실제 torchvision RoIAlign, PyTorch Attention, repository Encoder/Fusion,
실제 MMCV Attention/FFN/Norm, MMEngine/pycocotools COCO lifecycle, Detection Loss와
Hungarian 비용·assignment를 사용한다. BERT/Backbone 및 Image Deformable Attention은
소형 fixture로 대체한다. 전체 18개 config와 기존 Decoder·Loss·학습 Hook의 보존을
검사한다. 신규 테스트는 Dense Mask와 분할 Attention의 출력/Gradient 동등성,
Gamma=0, 기본 LN/Gamma 값, 초기 projection/RNG 보존, 새 parameter의 Detection Loss Gradient와 Optimizer 갱신, Iteration별 재추출·평가 중 재사용·
Resume/Test 재추출, Entropy 기준값, 통계 샘플링·DDP 합산과 로그 크기를 포함한다.

실제 CUDA 전체 모델은 MMCV CUDA ops, BERT, Pretrained Weight와 실제 Support
데이터가 있는 환경에서 다음 명령으로 FP32/AMP를 각각 검사한다.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/check_visual_tokens.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py --profile
# 같은 명령에 --amp를 추가하여 AMP도 확인한다.
```

학습된 Checkpoint 검사에는 `--checkpoint <path>`를 추가한다.
이 도구는 합성 Query의 실제 Detection Loss/Backward와 predict, 평가 Memory 재사용과 Checkpoint 재추출,
projection/Gamma Gradient를 검사한다. `--profile`은 동일 모델에서 Support on/off의
Forward+Backward 시간과 CUDA peak allocated/increment MiB를 보고한다. Warmup 1회 후
기본 3회를 측정하며 Optimizer Step이나 Weight 저장은 하지 않는다. 프로파일의 off도
같은 모델을 사용하며, Support on의 시간과 메모리에는 매 Training Forward의 전체
Support 재추출 비용도 포함한다.

로컬 검증 환경은 Windows / Python 3.13 / PyTorch 2.7.1+cpu / torchvision 0.22.1+cpu,
테스트 전용 MMEngine 0.10.7 / mmcv-lite 2.2.0 / pycocotools 2.0.11이다.
CUDA 전체 모델, 실제 Support 데이터 기반 학습·추론, GPU peak/처리량과 mAP는
로컬에서 실행하지 않았다. Gloo 프로세스 종료 검증은 CUDA/NCCL 검증을 대체하지 않는다.

Support 재추출 정책은 그대로 유지한다.
매 Iteration의 현재 Weight 반영, Validation 시작 시 재추출 및 평가 동안 재사용,
Best Checkpoint·Resume 반영, no_grad·혼합 submodule Mode 복원과 추출 실패 처리를
검증한다. 이번 변경은 Support K/V 전용 LN과 Gamma 초기값이며, 추출 경로·클래스 Mapping·
진단 통계 구조·학습 설정 및 활성 ACL Hook의 정책을 유지한다. CUDA 전체 모델 검증은 이 CPU 환경에서 수행하지 않는다.

Stage 정책 테스트는 실제 MMEngine OptimWrapper/파라미터별 Group,
ReduceOnPlateau Scheduler와 AdamW를 사용한다. LR=0의 FE Weight 불변과 Gradient
유지, Support LN/Gamma·나머지 학습 모듈의 실제 갱신, 원본 ACL LR 정책과의 동등성,
절반 LR 전환·patience, Stage별 Resume 및 실제 Detection Loss에서 고정 FE를
통과하는 Support Gradient를 검사한다.

이번 변경의 회귀 검증에는 고정 seed 및 DINO Xavier 이후 projection/RNG 보존,
K/V 전용 LN과 원본 Text residual, 기본 LN/Gamma 값, Layer별 독립성, 실제 Attention과
entropy 일치, LN/Gamma의 Stage 1 갱신과 기존 FE 동결, Stage 2/Resume 정책을 포함한다.
PyTorch/MMEngine의 새 checkpoint round-trip과 과거/부분 Support checkpoint 거부도 검사한다.
