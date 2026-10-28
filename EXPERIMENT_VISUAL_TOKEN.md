# ACL Support Visual Tokens

시작점은 `acl-serial-visual-text-contrastive`의 `449e69a`이다. Region-Text 보조
손실을 제거하고 HED 없는 sequential ACL에 Support Visual Token을 연결한다.
기존 순차 Decoder, DN, Query augmentation, optimizer, scheduler, checkpoint 및
Progressive FT hook은 유지한다. Stage 1부터 모든 그룹을 설정된 LR로 학습하는
시작 브랜치의 정책도 유지한다. 새 학습 parameter나 보조 Loss는 없다.

## Support와 Token 구성

- 18개 finetuning config가 각 Shot의 `train_dataloader.dataset`과 동일한
  annotation·이미지 경로·클래스 목록으로 별도 Support dataset을 구성한다.
  Query sampler 및 이미지 크기 필터는 적용하지 않는다. COCO loader가 원래
  제외하는 invalid/ignore/crowd annotation은 일반 GT instance에 포함하지 않는다.
- Support는 LoadImage → **LoadAnnotations → 고정 Resize(800,1333)** → Pack 순서다.
  같은 data preprocessor로 RGB 변환·정규화·padding을 수행한다. resized GT xyxy를
  그대로 사용한다. 평가용 annotation은 Prototype 생성에 사용하지 않는다.
- 매 Training Iteration에서 각 rank가 **전체 Support**를 batch size 1, worker 0,
  순차 순회한다. DataLoader 전용 Generator가 Query RNG를 소모하지 않는다.
- `no_grad()`에서 Backbone만 임시 eval로 바꾸고 원래 submodule mode를 복구한다.
  정규화된 Swin-B Stage 1 반환값 `norm1`(stride 8, 256 channels), **Neck 이전**에
  RoIAlign(7×7, aligned=True, sampling_ratio=2, spatial_scale=1/8) → GAP를 적용한다.
  FP32 instance 합계/개수로 class 평균을 만든다. 이미지 평균, projection,
  L2 정규화는 추가하지 않는다. Query Backbone gradient는 유지한다.
- Validation/Test의 첫 predict에서 현재 가중치로 전체 Support를 한 번 계산한다.
  평가 시작, train/eval 전환 및 weight 재로딩은 cache를 무효화한다.
  Prototype과 통계는 state_dict에 저장하지 않는다.
- BERT → text_feat_map **이후**, FE **이전**에 `[CLS] name1 <V1> . name2 <V2> . [SEP]`
  형태로 삽입한다. Visual Token은 해당 이름/period의 attention 블록에 포함되고
  block 내부 position ID를 차례로 부여한다. 기존 특수 Token의 FE/Decoder
  attention 연결과 padding 제외는 유지한다.
- 모든 Query는 전체 클래스 Prototype을 사용한다. 클래스 순서와 이름, COCO
  category→label 순서, 비어 있는 class, token 길이 초과를 검증한다.
  Visual 기능에서는 chunked/클래스 누락/비정형 recognition prompt를 지원하지 않는다.

## 분류와 Query Selection

클래스 c, Query q에 대해 기존 contrastive head의 Token Logit z를 사용한다.

```text
text_score[q,c]   = mean(sigmoid(z[q,t]))  # c의 이름 Token만
visual_score[q,c] = sigmoid(z[q,Vc])
fused_score[q,c]  = max(text_score[q,c], visual_score[q,c])
```

학습·추론 모두 동일한 Fusion이다. Focal Loss와 Hungarian Matching에는 위
Score와 동등한 log-odds를 전달한다. Text log-odds는
`logsumexp(logsigmoid(z_text)) - logsumexp(logsigmoid(-z_text))`로 계산한다.
Sigmoid 후 clipping·역변환을 피하고 FP32로 계산한다. 정확한 동점은 Text 경로다.
GT target은 클래스 one-hot이며 기존 분류 mask를 클래스 축으로 연결한다.
모든 Decoder layer, DN, 선택된 Encoder Query의 기존 Detection Loss에 적용한다.
L1/GIoU·loss weight·평균 분모는 변경하지 않는다.

Encoder Top-K 순위는 **개별 Token Logit Max**이다. 이름 Token과 Visual Token만
포함하고 CLS/SEP/period/PAD 등은 제외한다. Fusion한 클래스 평균으로 Query를
선택하지 않는다. 추론의 마지막 Decoder 결과는 이미 클래스별 Fusion Score이므로
원래 이름 Token 평균을 다시 적용하지 않는다.

## 로깅

기본 활성화된 `VisualTokenHook`은 Training에서 **Epoch 마지막 Batch만** 측정한다.
Validation/Test는 전체 dataset을 합산해 평가 종료 시 한 번 기록한다.
rank 0 일반 로그와 `work_dir/visual_token_stats.jsonl`에 Dataset/Shot/Epoch/Iteration,
전체 및 클래스별 `{value, count}`를 기록한다. 분모가 0이면 value는 null이다.

| 항목 | 정의 |
|---|---|
| win_all | 마지막 Decoder 일반 Query의 최고 Fusion 클래스에서 Visual 승리 비율 |
| win_gt | 같은 Query의 기존 Hungarian GT 매칭 클래스에서 Visual 승리 비율 |
| win_background | 미매칭 Query의 최고 Fusion 클래스에서 Visual 승리 비율 |
| text_score / visual_score / score_difference | 일반 Query×Class의 평균 Score와 Visual−Text |
| selection_visual | Encoder Top-K의 argmax Token이 Visual인 비율. 클래스는 그 Token의 소속 |
| text_norm | FE 이전, text_feat_map 이후 이름 Token의 평균 L2 Norm |
| prototype_norm | FE 삽입 전 원본 class Prototype의 평균 L2 Norm |

DN 및 다른 Decoder layer는 통계에서 제외한다. Win 비교는 Fusion과 동일한
안정적 log-odds 비교이며 동점은 Text다. 전체·Background와 GT 통계는 사용하는
클래스가 다르므로 서로의 단순 가중합 관계로 해석하지 않는다.

Training 매칭은 기존 Loss 경로를 재사용한다. Validation에는 기존 Forward
출력으로 Hungarian 계산만 추가한다. 18개 평가 pipeline은 **Resize 이후** GT를
읽으므로, 진단용 예측 box를 원본 pixel 좌표로 복원해 GT와 비교한다.
Pipeline의 GT 좌표 정책을 변경한다면 이 진단 처리도 함께 변경해야 한다.

추가 모델 Forward나 sample별 tensor 파일 저장은 없다. Forward 중 일시적인
detached 승리 mask/예측 클래스만 매칭까지 유지하고 즉시 해제한다. Batch 밖에는
클래스별 합계·개수만 남는다. DDP는 합계·개수를 합산하며, unshuffled
DefaultSampler의 평가 padding 중복은 집계에서 제외한다. 로깅 비활성화는
`custom_hooks`의 `VisualTokenHook.enabled=False`로 설정한다.

## 실행·검증

기존 실행 방법을 사용한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1
python -B -m unittest discover -s tests -v
```

일반 ACL 비교는 `--cfg-options model.support_visual_cfg=None`을 사용할 수 있다.
이때 Visual hook도 자동으로 통계를 생성하지 않는다. 새 parameter가 없으므로
기존 serial ACL/사전학습 checkpoint를 그대로 읽을 수 있다.

전체 CUDA 모델 smoke test는 MMCV, BERT, checkpoint 및 해당 Shot의 실제 Support
이미지·annotation이 있는 환경에서 실행한다. Synthetic Query 한 Batch로 실제
Loss/backward와 predict를 검사하며 optimizer step이나 checkpoint 저장은 없다.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/check_visual_tokens.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py
```

CPU 검증은 실제 torchvision RoIAlign, token mask/position 이동, Score Fusion,
gradient, cache와 state_dict, 기존 serial 메서드와의 disabled 경로 비교를 포함한다.
소형 fixture에서 실제 repository의 head Loss 라우팅, Focal/L1/GIoU, 모든 Decoder/DN/
Encoder target, Hungarian 비용과 scipy assignment를 실행한다. Logging on/off의
Loss·gradient 및 별도 optimizer update/RNG 동일성을 확인한다. Attention/BERT는
소형 fixture이므로 전체 모델 CUDA 검증을 대체하지 않는다. MMEngine 0.10.7로
18개 config를 파싱하고 Query 학습 설정이 시작 커밋과 동일함을 확인한다.

로컬 검증 결과: **36개 테스트 모두 통과**. CPU Gloo에서는 별도 단일 rank
검증으로 기본 group과 보조 group의 실제 통신 및 공통 cleanup을 확인했다.
Windows 파일 rendezvous는 DistStoreError가 발생해 로컬 TCPStore로 검증했다.
NCCL 또는 native allocator abort의 재현 검증은 포함하지 않는다.

2026-10-28 Support 초기화 오류 수정: `CocoDataset.load_data_list()`는 초기화 후
임시 `coco` 속성을 삭제한다. 이 속성을 참조하던 Support category 검증이 첫
iteration에서 AttributeError를 발생시켰다. Dataset의 resolved annotation 경로와
backend_args로 category 정보를 읽고, 유지되는 cat_ids 순서로 검증하도록 수정했다.
파일 읽기는 Support loader 생성 시 한 번이며 Prototype 재계산 정책은 유지한다.
기존 커밋의 동일 오류를 재현한 뒤, 실제 repository COCO 파싱 메서드·pycocotools·
MMEngine BaseDataset 초기화/직렬화와 순차 DataLoader를 통과하는 회귀 테스트를
추가했다. 이미지 변환/Backbone은 소형 fixture이며 실제 CUDA 학습 재실행은 미검증이다.

## 종료 처리와 남은 제한

Train/Test 진입점은 `9687696`의 Distributed Cleanup을 사용한다. Runner 생성부터
종료까지 try/finally로 감싸 모든 process group을 해제하고, 실패한 peer를 기다리는
barrier는 추가하지 않는다. Cleanup 실패가 원래 예외를 덮지 않는다.
`faulthandler`와 Runner 완료·group 정리 메시지로 종료 위치를 구분한다.
MMEngine의 warn_only 지원 여부는 실행 중 확인하고, stock 0.10.7에는 Runner 생성
후 동일한 warn_only 결정을 적용한다. 설치된 패치 버전도 지원한다.

`ChildFailedError`는 worker 실패 보고이며 native allocator 오류의 원인을 특정하지
않는다. Cleanup은 `munmap_chunk()`의 원인이 NCCL이었다는 증명이 아니고 SIGABRT를
억제하지도 않는다. 실제 재발 시 전체 로그와 native backtrace가 필요하다.

현재 로컬은 PyTorch 2.7.1+cpu / torchvision 0.22.1+cpu이고 MMCV/CUDA가 없다.
실제 Support dataset을 사용한 GPU 학습, NCCL 종료, mAP 및 처리량은 미검증이다.
매 iteration 전체 Support Backbone 계산은 Support 크기에 비례해 학습 시간을 늘린다.
