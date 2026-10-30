# ACL Support Local Descriptor Cross-Attention

기준은 `acl-vistoken-support-memory`의 `b8849b3`이다.
새 브랜치는 `codex/acl-support-local-cross-attn`이다. 기준의 HED 없는 순차 ACL
Decoder, DN, Query 증강, Detection Loss, Optimizer/Scheduler, 18개 Dataset/Shot
설정 및 분산 프로세스 종료 처리를 유지한다.

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

Q는 Fusion 직후 Text, K/V는 해당 클래스의 Local Descriptor다. 각 Forward 안에서는 변경하지 않는다.
클래스 이름 Token과 그 period만 같은 클래스 Memory를 볼 수 있고 CLS/SEP/Padding은
Support residual을 받지 않는다. 여러 Batch의 같은 클래스 Query를 묶어 Attention을
계산한다. Query끼리 Attention하지 않으므로 Dense Class-wise Mask와 수학적으로
동일하다. 모든 클래스 descriptor를 사용하면서 전체 B×Head×T×S Mask를 피한다.

Attention은 기존 Text Attention과 동일한 차원/Head 수, Dropout=0으로 구성한다.
Q/K/V/출력 projection은 일반 초기화이고, Channel-wise Gamma는 `1e-4`로 초기화한다.
기존 DINO Xavier 초기화는 2차원 이상 Weight만 변경하므로 Gamma 초기값을 보존한다.
추가 LayerNorm, 위치 인코딩, Target–Support Attention은 없다. 일반 Forward는
`need_weights=False`로 SDPA 경로를 사용하고 Support는 detach한다.

Text Self-Attention의 K/V 결합과 공통 DETR Layer의 explicit K/V 확장은 제거했다.
Support 활성 경로의 Text Mask는 Batch×Head 순서로 repeat_interleave한다. Support
비활성 경로는 원래 repeat 동작까지 그대로 보존한다. 클래스가 다른 Support 변경의
직접 영향은 Cross-Attention에서 차단한다. 이후 정상 Image–Text Fusion을 통한
간접 영향은 모델의 원래 동작이다.

새 parameter는 `encoder.support_layers.*` 아래에만 추가한다. 기존 Weight의 Key와
Shape를 유지하며, 기존 Pretrained 로딩에서 새 parameter만 누락될 수 있다.
`support_visual_cfg=None`이면 새 모듈을 만들지 않고 기존 Forward/Backward를 따른다.

## 경량 통계: schema_version 3

`visual_token_stats.jsonl`과 기존 text_norm, support_token_norm,
enhanced_text_norm을 유지한다. support_token_count는 이제 Local Descriptor 수다.
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

기존 실행 명령을 사용한다. 별도 Support 디스크 캐시 없이 실행한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1
python -B -m unittest discover -s tests -v
```

CPU 검증은 실제 torchvision RoIAlign, PyTorch Attention, repository Encoder/Fusion,
실제 MMCV Attention/FFN/Norm, MMEngine/pycocotools COCO lifecycle, Detection Loss와
Hungarian 비용·assignment를 사용한다. BERT/Backbone 및 Image Deformable Attention은
소형 fixture로 대체한다. 전체 18개 config와 기존 Decoder·Loss·학습 Hook의 보존을
검사한다. 신규 테스트는 Dense Mask와 분할 Attention의 출력/Gradient 동등성,
Gamma=0, 새 parameter의 Detection Loss Gradient와 Optimizer 갱신, Iteration별 재추출·평가 중 재사용·
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

재추출 방식 변경 후 로컬 전체 회귀 테스트: **64개 통과, 실패 0개, 건너뜀 0개**.
매 Iteration의 현재 Weight 반영, Validation 시작 시 재추출 및 평가 동안 재사용,
Best Checkpoint·Resume 반영, no_grad·혼합 submodule Mode 복원과 추출 실패 처리를
검증했다. Cross-Attention·Gamma·클래스 Mapping·진단 통계 및 학습·평가 설정은
기존 구현을 유지한다. CUDA 전체 모델 검증은 이 CPU 환경에서 수행하지 않았다.
