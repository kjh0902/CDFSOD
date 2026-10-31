# ACL Neck-based Pretrained Support Fusion

브랜치: `codex/acl-support-neck-pretrained-fusion` · 기준: `944d0d1`.
기존 serial ACL decoder, DN, query augmentation, detection loss와 18개 dataset/shot의
학습 설정을 유지한다. 일반 초기화 비교 모드는 제공하지 않는다.

## Support memory

해당 shot의 학습 annotation과 이미지만 사용한다. 고정 Resize(800,1333),
기존 정규화/padding 뒤 현재 backbone → neck의 **Feature 0** (stride 8, 256 channels)에
GT xyxy RoIAlign을 적용한다. 7×7, aligned=True, sampling_ratio=2, spatial_scale=1/8이다.
객체별 49개 descriptor를 클래스별로 모두 연결한다. Selection, aggregation,
projection, L2 정규화, shot별 자르기/반복은 없다. Memory는 LN 이전 FP32 descriptor다.

매 학습 forward에서 전체 support를 현재 backbone/neck으로 다시 추출한다.
추출은 no_grad이며 backbone과 neck의 모든 하위 모듈을 임시 eval로 설정하고,
성공/실패 모두 원래의 혼합 train/eval mode로 복원한다. Query 경로의 학습은 유지한다.
Support DataLoader는 독립 RNG를 사용하며 GT category 순서/invalid/ignore/crowd 처리를 유지한다.

Validation/test 시작 hook에서 현재 로드된 weight로 한 번 추출하여 평가 동안 재사용한다.
각 rank는 전체 support를 독립 추출한다. Logging을 꺼도 동일하게 동작한다.
train 전환과 checkpoint 로드는 평가 memory를 무효화한다. Memory는 checkpoint나 디스크에
저장하지 않으며, 과거 디스크 cache를 읽지 않는다.

## Layer별 독립 attention

각 FE layer의 실행 순서는 기존 target image–text fusion → support-to-text attention →
기존 text self-attention/FFN → image deformable self-attention/FFN이다.

Support layer i는 대응 FE fusion layer i의 다음 값을 독립적으로 복사한다.

| Support parameter | Pretrained FE parameter |
|---|---|
| text_norm | layer_norm_l |
| support_norm | layer_norm_v |
| attn.q_proj | attn.l_proj |
| attn.k_proj | attn.v_proj |
| attn.v_proj | attn.values_v_proj |
| attn.out_proj | attn.out_l_proj |
| gamma | gamma_l |

각 projection/LN의 weight와 bias를 모두 복사한다. 입출력 256, attention 내부 1,024,
4 heads × 256이다. Temperature=1, dropout=0이며 pretrained gamma에 추가 배율을 적용하지 않는다.
기존 FE는 image×text 점수 행렬을 전치하여 text를 갱신하므로 l_proj가 Q, v_proj가 K에 대응한다.
Normal-range FP32에서 이 방향의 연산과 동등하다. 새 경로는 안정적인 SDPA를 사용한다.

Tn=LN_text(T), Sn=LN_support(S), delta=Attention(Tn,Sn,Sn), output=T+gamma*delta.
**Residual 기준은 정규화 전 T**다. Gamma=0이면 support 모듈은 입력을 보존한다.
같은 LN 이전 memory를 여섯 layer가 각각 다른 LN으로 처리한다. Stop-gradient는 LN 이전에
적용되어 복사된 LN도 학습된다. 같은 클래스 descriptor만 읽으며 클래스명과 period만 갱신한다.
CLS/SEP/padding은 갱신하지 않는다. 원본 FE/다른 support layer와 parameter storage를 공유하지 않는다.

## Checkpoint와 학습 단계

최초 pretrained state dict에 support 키가 없을 때만, 대응 FE key에서 support key를 생성한다.
이 과정은 PyTorch와 MMEngine의 기존 checkpoint 로딩 경로 안에서 수행되어 DINO 초기화나
checkpoint 로딩에 의해 복사 값이 나중에 덮어써지지 않는다. Layer별 초기화 결과를 로그에 남긴다.
새 support key가 모두 있는 checkpoint는 그대로 복원하며 resume/test에서 재이식하지 않는다.
일부 support key만 있거나 이전 256차원 support 구조이면 명시적 오류로 중단한다.
필수 FE source key가 없거나 shape가 맞지 않아도 중단하며 랜덤 초기화로 조용히 진행하지 않는다.
이전 support checkpoint 자동 변환 및 optimizer resume 호환은 제공하지 않는다.

| Group | Stage 1 | Stage 2 시작 |
|---|---|---|
| 기존 FE/ACL Other | LR=0 | 당시 head LR로 학습 시작 |
| 새 support projection, 양쪽 LN, gamma | 설정된 LR로 학습 | 당시 head LR로 계속 학습 |
| Backbone/BERT/head 관련 | 기존 설정·scheduler | scheduler 유지 |

새 parameter는 encoder.support_layers.* 아래에 둔다. 그룹 추가·재정렬 없이 기존 paramwise
분류를 유지한다. BERT LR이 최초 LR의 절반 이하가 되면 다음 epoch 시작에 한 번 전환한다.
Patience Stage 1=3, Stage 2=8 및 resume 복원 정책을 유지한다. support_visual_cfg=None에서는
support 모듈/초기화를 사용하지 않고 기존 비활성 경로를 유지한다.

## Statistics schema 4

architecture=`support_neck_pretrained_cross_attention`.
기존 text_norm, support_token_norm, enhanced_text_norm과 layer별 entropy, residual, gamma를 유지한다.
support_token_norm은 이제 **neck/RoIAlign 이후, support LN 이전** norm이다.
text_norm은 FE 입력, enhanced_text_norm은 FE 마지막 출력이며 두 값을 residual 크기로 비교하지 않는다.

- normalized_attention_entropy: head/query별 -sum(p log p)/log(S_class) 평균.
- normalized_attention_entropy_per_head: 각 head의 value/count 목록.
- residual_contribution: ||gamma*delta|| / max(||T||,1e-12), T는 LN 이전.
- ungated_update_contribution: ||delta|| / max(||T||,1e-12), 같은 query 표본.
- gamma: mean_absolute, max_absolute, 모집단 std.

Entropy는 실제 forward의 LN 결과와 복사된 Q/K projection 및 head scaling을 사용한다.
FP32에서 key를 1,024개씩 나누어 정확한 entropy를 계산한다. S_class=1이면 0,
관측이 없으면 null/count=0이다. 학습 epoch 마지막 batch와 평가 batch 0/50/100/...에서
수집한다. Layer·class별 최대 8 query를 entropy에 사용하며 residual은 선택 batch의 모든
유효 class token을 사용한다. DDP padding 이미지를 제외하고 sum/count/head sum만 합산한다.
Gamma/support 수는 rank 0 snapshot이며 영구 통계는 O(classes+layers*heads)다.

## 실행과 검증

```bash
bash run_cdfsod.sh --dataset UODD --shot 10 --dry-run
bash run_cdfsod.sh --dataset UODD --shot 10
bash run_cdfsod.sh --dataset NEU-DET --shot 10
python analyze_results_cdfsod.py exp_cdfsod_results_neck_pretrained
python -B -m unittest discover -s tests -v
```

출력은 exp_cdfsod_results_neck_pretrained/<dataset>/<shot>shot/로 분리한다.
`--resume`은 이 브랜치에서 생성된 동일 구조 checkpoint에만 사용한다.

GPU 환경에서 전체 모델과 실제 support의 FP32/AMP 확인:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/check_visual_tokens.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_UODD_10shot.py --profile
# 동일 명령에 --amp를 추가하여 AMP도 확인한다.
# 학습된 새 checkpoint 검사에는 --checkpoint <path>를 추가한다.
```

이 도구는 pretrained 로딩, detection loss/backward, neck descriptor, evaluation cache,
통계 및 CUDA 시간/메모리를 점검한다. Optimizer step이나 weight 저장은 하지 않는다.
CPU 테스트는 실제 repository fusion/encoder/text attention, RoIAlign, PyTorch SDPA,
MMEngine checkpoint/optimizer/scheduler와 detection loss fixture를 사용한다.
전체 CUDA 모델, 실제 BERT/데이터/원본 pretrained checkpoint 검증을 대신하지 않는다.
